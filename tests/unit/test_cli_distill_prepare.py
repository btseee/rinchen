"""`rinchen distill prepare`: what it prints, plans and writes (the original corpora of D7, the
KD corpora, the validation files, the student's config and the manifest), the options and how
a rerun leaves the output directory. What it refuses, and the order in which it hashes, reads
and commits, is in `test_cli_distill_prepare_refusals.py`.

Nothing here starts Eole. The corpus, the KD file and the SentencePiece models are the toy's
(`tests.support.planted_student`), so the weights are those worked out there: at temperature
0.5, back-translation ratio 0.5 and weight total 100 the original corpora weigh 22 and 6
(English to Mongolian and back) and 15 and 7 (Russian), the KD corpus of a direction the sum
of the original weights for `--mix 1:1` and three times it for `3:1`. The toy file holds
5,200 rows (3,600 authentic, 1,200 back-translated, 400 dev) and the KD file 3,600.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from rinchen.corpus.writers import write_corpus
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.train.eole_cfg import read_config, validate_train_config
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.distill_cli import (
    TOY_OPTIONS,
    PrepareWorkspace,
    invoke,
    kd_with_exclusions,
    make_prepare_workspace,
    run_ok,
)
from tests.support.launch_cli import snapshot
from tests.support.planted_student import (
    BT_DIRECTORY,
    DIRECTIONS,
    KD_ROWS,
    KD_WEIGHTS,
    ORIGINAL_WEIGHTS,
    Inputs,
    build_inputs,
)

AUTH_ROWS = {
    f"{EN_MN}.auth": 1200,
    f"{MN_EN}.auth": 1200,
    f"{MN_RU}.auth": 600,
    f"{RU_MN}.auth": 600,
}
# Each direction has 300 back-translated rows: round(0.5 * authentic rows) is 600 or 300 and
# the corpus has 300.
BT_ROWS = {f"{direction}.bt": 300 for direction in DIRECTIONS}
ORIGINAL_ROWS = {
    name: AUTH_ROWS[f"{direction}.auth"] if name.endswith(".auth") else 300
    for direction in DIRECTIONS
    for name in (f"{direction}.auth", f"{direction}.bt")
}
# How many times more often a KD row is drawn than an original row of its direction, for
# `--mix 1:1` (28 / 1200 against 28 / 1500; 22 / 600 against 22 / 900), `3:1` (84 / 1200
# against 28 / 1500; 66 / 600 against 22 / 900) and `1:3` (28 / 1200 against 84 / 1500; 22 /
# 600 against 66 / 900).
REPEAT = {
    "1:1": {EN_MN: "1.250", MN_EN: "1.250", MN_RU: "1.500", RU_MN: "1.500"},
    "3:1": {EN_MN: "3.750", MN_EN: "3.750", MN_RU: "4.500", RU_MN: "4.500"},
    "1:3": {EN_MN: "0.417", MN_EN: "0.417", MN_RU: "0.500", RU_MN: "0.500"},
}
KD_REASONS = (
    "not_train",
    "filtered",
    "wrong_origin",
    "empty",
    "line_break",
    "unknown_direction",
)
KD_EXCLUSIONS = [f"kd excluded {reason}: 0" for reason in KD_REASONS]


def original_statistics() -> list[str]:
    return [
        "corpus files: 1",
        "kd files: 1",
        "rows read: 5200",
        "rows kept: 4800",
        "excluded not_train: 400",
        "excluded filtered: 0",
        "excluded excluded_origin: 0",
        "excluded empty: 0",
        "excluded line_break: 0",
        "excluded by origin ft: 0",
        "excluded by origin synth_llm: 0",
        "excluded by origin synth_nmt: 0",
        "duplicate ids: 0",
        *(f"dev lines {direction}: 100" for direction in DIRECTIONS),
        "dev in train: 0",
    ]


def plan_lines(*, mix: str = "1:1", tag: str = "ft") -> list[str]:
    """What a run prints before it writes, for the toy at the `TOY_OPTIONS`: the KD file
    has no excluded row; a `K:O` mix weighs each KD corpus `K` times the sum of its
    direction's original weights and each original corpus `O` times its own."""
    kd_times, original_times = (int(part) for part in mix.split(":"))
    return [
        *original_statistics(),
        "kd rows read: 3600",
        "kd rows kept: 3600",
        *KD_EXCLUSIONS,
        "kd duplicate ids: 0",
        *(f"kd rows {d}: {KD_ROWS[f'{d}.kd']}" for d in DIRECTIONS),
        f"mix: {mix}",
        f"kd tag: {tag}",
        *(
            f"corpus {name}: rows {ORIGINAL_ROWS[name]}, "
            f"weight {weight * original_times}"
            for name, weight in ORIGINAL_WEIGHTS.items()
        ),
        *(
            f"corpus {name}: rows {KD_ROWS[name]}, weight {weight * kd_times}"
            for name, weight in KD_WEIGHTS.items()
        ),
        *(f"repeat {d}.kd: {REPEAT[mix][d]}" for d in DIRECTIONS),
    ]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    return build_inputs(tmp_path_factory.mktemp("cli_distill_prepare_inputs"))


@pytest.fixture
def workspace(tmp_path: Path, inputs: Inputs) -> PrepareWorkspace:
    return make_prepare_workspace(tmp_path, inputs)


def manifest_of(workspace: PrepareWorkspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_of(workspace: PrepareWorkspace) -> dict[str, Any]:
    return yaml.safe_load(workspace.config.read_text(encoding="utf-8"))


# --- help and defaults ---------------------------------------------------------------------


def test_the_distill_group_lists_prepare_and_evaluate() -> None:
    result = invoke(["distill", "--help"])

    assert result.exit_code == 0
    assert "prepare" in result.stdout
    assert "evaluate" in result.stdout


def test_help_names_every_option_and_marks_exactly_the_five_required_ones() -> None:
    result = invoke(["distill", "prepare", "--help"])

    assert result.exit_code == 0
    for option in (
        "--corpus",
        "--kd",
        "--spm-model",
        "--config",
        "--out-dir",
        "--mix",
        "--no-kd-tag",
        "--tag-placement",
        "--bt-ratio",
        "--temperature",
        "--seed",
        "--dev-max-lines",
        "--weight-total",
        "--dry-run",
    ):
        assert option in result.stdout, option
    assert result.stdout.count("[required]") == 5
    assert "<src|tgt>" in result.stdout


def test_the_help_states_the_defaults() -> None:
    text = " ".join(invoke(["distill", "prepare", "--help"]).stdout.split())

    for option, default in (
        ("--mix", "1:1"),
        ("--tag-placement", "src"),
        ("--bt-ratio", "0.0"),
        ("--temperature", "1.0"),
        ("--seed", "1234"),
        ("--dev-max-lines", "1000"),
        ("--weight-total", "100"),
    ):
        assert (
            f"[default: {default}]" in text.split(f"{option} ", 1)[1].split(" --", 1)[0]
        ), option


def test_the_defaults_are_the_ones_the_help_states(
    workspace: PrepareWorkspace,
) -> None:
    plain = run_ok(workspace.args("--dry-run")).stdout
    explicit = run_ok(
        workspace.args(
            "--dry-run",
            "--mix",
            "1:1",
            "--tag-placement",
            "src",
            "--bt-ratio",
            "0.0",
            "--temperature",
            "1.0",
            "--seed",
            "1234",
            "--dev-max-lines",
            "1000",
            "--weight-total",
            "100",
        )
    ).stdout

    assert plain == explicit
    assert "corpus eng_Latn-khk_Cyrl.auth: rows 1200, weight 33" in plain
    assert "corpus eng_Latn-khk_Cyrl.kd: rows 1200, weight 33" in plain
    assert not any(".bt:" in line for line in plain.splitlines())


# --- the dry run -----------------------------------------------------------------------------


def test_a_dry_run_prints_the_statistics_the_weights_and_the_repeat_factors(
    workspace: PrepareWorkspace,
) -> None:
    result = run_ok(workspace.args(*TOY_OPTIONS, "--dry-run"))

    assert result.stdout.splitlines() == [*plan_lines(), "nothing was written"]


def test_a_dry_run_writes_nothing_not_even_the_output_directory(
    workspace: PrepareWorkspace,
) -> None:
    before = snapshot(workspace.root)

    run_ok(workspace.args(*TOY_OPTIONS, "--dry-run"))

    assert not workspace.out_dir.exists()
    assert snapshot(workspace.root) == before


def test_several_corpus_files_are_read_as_one_and_counted(
    tmp_path: Path, inputs: Inputs
) -> None:
    workspace = make_prepare_workspace(tmp_path, inputs)
    frame = pl.read_parquet(inputs.corpus)
    training = write_corpus(
        frame.filter(pl.col("split") != "dev"), tmp_path / "train.parquet"
    )
    development = write_corpus(
        frame.filter(pl.col("split") == "dev"), tmp_path / "dev.parquet"
    )
    assert (training, development) == (4800, 400)

    result = run_ok(
        workspace.args(
            *TOY_OPTIONS,
            "--dry-run",
            corpora=[tmp_path / "train.parquet", tmp_path / "dev.parquet"],
        )
    )

    assert result.stdout.splitlines() == [
        "corpus files: 2",
        *plan_lines()[1:],
        "nothing was written",
    ]


def test_a_mix_of_three_to_one_weighs_every_kd_corpus_three_times_the_sum_of_its_directions_originals(
    workspace: PrepareWorkspace,
) -> None:
    result = run_ok(workspace.args(*TOY_OPTIONS, "--mix", "3:1", "--dry-run"))

    assert result.stdout.splitlines() == [
        *plan_lines(mix="3:1"),
        "nothing was written",
    ]
    assert "corpus eng_Latn-khk_Cyrl.kd: rows 1200, weight 84" in result.stdout
    assert "corpus rus_Cyrl-khk_Cyrl.kd: rows 600, weight 66" in result.stdout


def test_the_mix_is_reduced_by_its_greatest_common_divisor(
    workspace: PrepareWorkspace,
) -> None:
    reduced = run_ok(workspace.args(*TOY_OPTIONS, "--mix", "2:2", "--dry-run")).stdout

    assert reduced.splitlines() == [*plan_lines(), "nothing was written"]


def test_a_mix_that_favours_the_originals_scales_them(
    workspace: PrepareWorkspace,
) -> None:
    result = run_ok(workspace.args(*TOY_OPTIONS, "--mix", "1:3", "--dry-run"))

    assert result.stdout.splitlines() == [
        *plan_lines(mix="1:3"),
        "nothing was written",
    ]
    assert "corpus eng_Latn-khk_Cyrl.auth: rows 1200, weight 66" in result.stdout
    assert "corpus eng_Latn-khk_Cyrl.kd: rows 1200, weight 28" in result.stdout
    # The KD rows are drawn 1/3 as often relative to the originals: 28 / 1200 against
    # (66 + 18) / 1500.
    assert "repeat eng_Latn-khk_Cyrl.kd: 0.417" in result.stdout


def test_the_kd_counts_by_reason_are_printed_for_a_file_with_rows_that_do_not_train(
    tmp_path: Path, inputs: Inputs
) -> None:
    kd = kd_with_exclusions(tmp_path / "bad_kd", inputs)
    workspace = make_prepare_workspace(tmp_path / "w", inputs, kd=kd)

    result = run_ok(workspace.args(*TOY_OPTIONS, "--dry-run"))

    lines = result.stdout.splitlines()
    assert lines[lines.index("kd rows read: 3606") :][:10] == [
        "kd rows read: 3606",
        "kd rows kept: 3600",
        "kd excluded not_train: 1",
        "kd excluded filtered: 1",
        "kd excluded wrong_origin: 1",
        "kd excluded empty: 1",
        "kd excluded line_break: 1",
        "kd excluded unknown_direction: 1",
        "kd duplicate ids: 0",
        f"kd rows {EN_MN}: 1200",
    ]
    assert "corpus eng_Latn-khk_Cyrl.kd: rows 1200, weight 28" in lines


# --- a real run ------------------------------------------------------------------------------


def test_a_run_prints_the_plan_then_what_it_wrote(workspace: PrepareWorkspace) -> None:
    result = run_ok(workspace.args(*TOY_OPTIONS))

    assert result.stdout.splitlines() == [
        *plan_lines(),
        "corpora: 12 (reused 0)",
        "validation lines: 400",
        f"config: {workspace.out_dir}/config.yaml",
        f"manifest: {workspace.out_dir}/manifest.json",
    ]


def test_a_run_writes_the_corpora_the_validation_files_the_config_and_the_manifest(
    workspace: PrepareWorkspace,
) -> None:
    run_ok(workspace.args(*TOY_OPTIONS))

    files = sorted(workspace.written())
    assert files == sorted(
        [
            "config.yaml",
            "manifest.json",
            "valid.src",
            "valid.tgt",
            *(
                f"data/auth/{direction}.auth.{extension}"
                for direction in DIRECTIONS
                for extension in ("src", "tgt", "json")
            ),
            *(
                f"{BT_DIRECTORY}/{direction}.bt.{extension}"
                for direction in DIRECTIONS
                for extension in ("src", "tgt", "json")
            ),
            *(
                f"data/kd/{direction}.kd.{extension}"
                for direction in DIRECTIONS
                for extension in ("src", "tgt", "json")
            ),
        ]
    )
    assert not (workspace.out_dir / "run").exists()


def test_the_config_validates_names_no_checkpoint_to_start_from_and_carries_the_kd_tag(
    workspace: PrepareWorkspace,
) -> None:
    run_ok(workspace.args(*TOY_OPTIONS))

    config = config_of(workspace)
    validate_train_config(read_config(workspace.config, "config"))
    assert "train_from" not in str(config)
    assert "reset_optim" not in str(config)
    assert config["seed"] == 1234
    assert config["src_vocab"] == f"{workspace.out_dir}/run/vocab.src"
    assert config["training"]["model_path"] == f"{workspace.out_dir}/run/ckpt"
    assert config["src_vocab_size"] == 641
    data = config["data"]
    assert sorted(data) == sorted([*ORIGINAL_WEIGHTS, *KD_WEIGHTS, "valid"])
    for name, weight in {**ORIGINAL_WEIGHTS, **KD_WEIGHTS}.items():
        assert data[name]["weight"] == weight
    for name in KD_WEIGHTS:
        assert data[name]["transforms"] == ["sentencepiece", "prefix", "filtertoolong"]
        assert data[name]["src_prefix"].endswith(" <ft>")
        assert data[name]["tgt_prefix"] == ""
    assert data[f"{EN_MN}.auth"]["src_prefix"] == "<2khk_Cyrl>"
    assert data[f"{EN_MN}.kd"]["src_prefix"] == "<2khk_Cyrl> <ft>"
    assert data["valid"]["path_src"] == f"{workspace.out_dir}/valid.src"


def test_the_manifest_vouches_for_the_inputs_the_outputs_and_the_numbers_of_the_run(
    workspace: PrepareWorkspace,
) -> None:
    run_ok(workspace.args(*TOY_OPTIONS))

    manifest = manifest_of(workspace)
    assert manifest.stage == "distill_prepare"
    assert manifest.seed == 1234
    assert manifest.config_sha256 == sha256(workspace.base)
    assert manifest.status == {
        "prepare": "ok",
        "tag_placement": "src",
        "mix": "1:1",
    }
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (str(path), sha256(path)) for path in workspace.inputs
    ]
    assert manifest.inputs[2].path == str(workspace.kd)
    assert manifest.inputs[3].path == str(workspace.kd.parent / "manifest.json")
    out = workspace.out_dir
    assert [(Path(item.path), item.rows) for item in manifest.outputs[:3]] == [
        (out / "config.yaml", None),
        (out / "valid.src", 400),
        (out / "valid.tgt", 400),
    ]
    for item in manifest.outputs:
        assert item.sha256 == sha256(Path(item.path)), item.path
    corpus_files = [item for item in manifest.outputs[3:] if item.path.endswith("src")]
    assert {
        Path(item.path).name.removesuffix(".src"): item.rows for item in corpus_files
    } == ORIGINAL_ROWS | KD_ROWS
    assert len(manifest.outputs) == 3 + 12 * 3
    assert "manifest.json" not in " ".join(item.path for item in manifest.outputs)


def test_the_manifest_metrics_are_the_options_the_weights_the_repeat_factors_and_every_count(
    workspace: PrepareWorkspace,
) -> None:
    run_ok(workspace.args(*TOY_OPTIONS))

    metrics = manifest_of(workspace).metrics
    expected = {
        "mix.kd": 1.0,
        "mix.original": 1.0,
        "kd_tag": 1.0,
        "bt_ratio": 0.5,
        "temperature": 0.5,
        "weight_total": 100.0,
        "valid.lines": 400.0,
        "count.rows_read": 5200.0,
        "count.rows_kept": 4800.0,
        "count.excluded.not_train": 400.0,
        "count.excluded.filtered": 0.0,
        "count.excluded.excluded_origin": 0.0,
        "count.excluded.empty": 0.0,
        "count.excluded.line_break": 0.0,
        "count.excluded_by_origin.ft": 0.0,
        "count.duplicate_ids": 0.0,
        "count.dev.rows_usable": 400.0,
        "count.dev.dev_in_train": 0.0,
        "kd.count.rows_read": 3600.0,
        "kd.count.rows_kept": 3600.0,
        "kd.count.duplicate_ids": 0.0,
        "kd.count.rows_by_label.ft:toy": 3600.0,
        "kd.repeat.eng_Latn-khk_Cyrl": 1.25,
        "kd.repeat.khk_Cyrl-eng_Latn": 1.25,
        "kd.repeat.khk_Cyrl-rus_Cyrl": 1.5,
        "kd.repeat.rus_Cyrl-khk_Cyrl": 1.5,
    }
    for name, value in expected.items():
        assert metrics[name] == value, name
    for reason in KD_REASONS:
        assert metrics[f"kd.count.excluded.{reason}"] == 0.0, reason
    for name, weight in {**ORIGINAL_WEIGHTS, **KD_WEIGHTS}.items():
        assert metrics[f"weight.{name}"] == float(weight), name
    for name, rows in ORIGINAL_ROWS.items():
        assert metrics[f"count.rows_by_corpus.{name}"] == float(rows), name
    for name, rows in KD_ROWS.items():
        direction = name.removesuffix(".kd")
        assert metrics[f"kd.count.rows_by_direction.{direction}"] == float(rows)
        assert f"count.rows_by_corpus.{name}" not in metrics
    assert json.loads(json.dumps(metrics)) == metrics


def test_the_kd_counts_of_a_file_with_rows_that_do_not_train_are_in_the_metrics(
    tmp_path: Path, inputs: Inputs
) -> None:
    kd = kd_with_exclusions(tmp_path / "bad_kd", inputs)
    workspace = make_prepare_workspace(tmp_path / "w", inputs, kd=kd)

    run_ok(workspace.args(*TOY_OPTIONS))

    metrics = manifest_of(workspace).metrics
    assert metrics["kd.count.rows_read"] == 3606.0
    assert metrics["kd.count.rows_kept"] == 3600.0
    for reason in KD_REASONS:
        assert metrics[f"kd.count.excluded.{reason}"] == 1.0, reason
    assert "kd.count.rows_by_direction.eng_Latn-rus_Cyrl" not in metrics


# --- a rerun ---------------------------------------------------------------------------------


def test_a_rerun_reuses_every_corpus_and_rewrites_the_manifest(
    workspace: PrepareWorkspace,
) -> None:
    run_ok(workspace.args(*TOY_OPTIONS))
    first = manifest_of(workspace)
    files_before = {
        name: content
        for name, content in workspace.written().items()
        if name != "manifest.json"
    }

    result = run_ok(workspace.args(*TOY_OPTIONS))

    assert "corpora: 12 (reused 12)" in result.stdout.splitlines()
    second = manifest_of(workspace)
    assert second.created != first.created
    assert second.outputs == first.outputs
    assert second.inputs == first.inputs
    assert {
        name: content
        for name, content in workspace.written().items()
        if name != "manifest.json"
    } == files_before


def test_a_rerun_with_another_mix_changes_only_the_config_and_the_manifest(
    workspace: PrepareWorkspace,
) -> None:
    run_ok(workspace.args(*TOY_OPTIONS))
    before = workspace.written()

    result = run_ok(workspace.args(*TOY_OPTIONS, "--mix", "3:1"))

    after = workspace.written()
    changed = sorted(name for name in before if before[name] != after[name])
    assert changed == ["config.yaml", "manifest.json"]
    assert "corpora: 12 (reused 12)" in result.stdout.splitlines()
    assert config_of(workspace)["data"][f"{EN_MN}.kd"]["weight"] == 84
    assert manifest_of(workspace).status["mix"] == "3:1"


# --- the tag, the placement, the student's model ---------------------------------------------


def test_no_kd_tag_leaves_the_plain_direction_tag_and_says_so(
    workspace: PrepareWorkspace,
) -> None:
    result = run_ok(workspace.args(*TOY_OPTIONS, "--no-kd-tag"))

    assert "kd tag: none" in result.stdout.splitlines()
    data = config_of(workspace)["data"]
    assert data[f"{EN_MN}.kd"]["src_prefix"] == "<2khk_Cyrl>"
    assert data[f"{MN_EN}.kd"]["src_prefix"] == "<2eng_Latn>"
    assert manifest_of(workspace).metrics["kd_tag"] == 0.0


def test_the_kd_tag_is_on_by_default(workspace: PrepareWorkspace) -> None:
    result = run_ok(workspace.args(*TOY_OPTIONS))

    assert "kd tag: ft" in result.stdout.splitlines()
    assert manifest_of(workspace).metrics["kd_tag"] == 1.0


def test_the_target_placement_puts_the_kd_tag_alone_in_the_source_prefix(
    workspace: PrepareWorkspace,
) -> None:
    run_ok(workspace.args(*TOY_OPTIONS, "--tag-placement", "tgt"))

    data = config_of(workspace)["data"]
    assert data[f"{EN_MN}.kd"]["src_prefix"] == "<ft>"
    assert data[f"{EN_MN}.kd"]["tgt_prefix"] == "<2khk_Cyrl>"
    assert data[f"{EN_MN}.auth"]["src_prefix"] == ""
    assert data[f"{EN_MN}.auth"]["tgt_prefix"] == "<2khk_Cyrl>"
    assert manifest_of(workspace).status["tag_placement"] == "tgt"


def test_a_student_with_another_sentencepiece_model_gets_its_own_vocabulary_size_and_validation(
    tmp_path: Path, inputs: Inputs
) -> None:
    standard = make_prepare_workspace(tmp_path / "standard", inputs)
    small = make_prepare_workspace(tmp_path / "small", inputs)
    run_ok(standard.args(*TOY_OPTIONS))

    run_ok(small.args(*TOY_OPTIONS, spm_model=inputs.spm500))

    assert config_of(small)["src_vocab_size"] == 501
    assert config_of(standard)["src_vocab_size"] == 641
    assert small.written()["valid.src"] != standard.written()["valid.src"]
    assert (
        small.written()["data/kd/eng_Latn-khk_Cyrl.kd.src"]
        == (standard.written()["data/kd/eng_Latn-khk_Cyrl.kd.src"])
    )
    assert manifest_of(small).inputs[-1].path == str(inputs.spm500)
    assert manifest_of(small).inputs[-1].sha256 == sha256(inputs.spm500)


def test_a_back_translation_ratio_of_zero_leaves_out_the_bt_corpora(
    workspace: PrepareWorkspace,
) -> None:
    result = run_ok(
        workspace.args(
            "--temperature",
            "0.5",
            "--dev-max-lines",
            "100",
            "--bt-ratio",
            "0",
        )
    )

    assert "corpora: 8 (reused 0)" in result.stdout.splitlines()
    assert not (workspace.out_dir / BT_DIRECTORY.rsplit("/", 1)[0]).exists()
    data = config_of(workspace)["data"]
    assert not any(name.endswith(".bt") for name in data)
    # The KD weight follows the originals' sum, whatever they are.
    for direction in DIRECTIONS:
        assert data[f"{direction}.kd"]["weight"] == data[f"{direction}.auth"]["weight"]


def test_the_options_reach_the_run(workspace: PrepareWorkspace) -> None:
    result = run_ok(
        workspace.args(
            *TOY_OPTIONS[:4],
            "--dev-max-lines",
            "7",
            "--weight-total",
            "50",
            "--seed",
            "99",
        )
    )

    assert "validation lines: 28" in result.stdout.splitlines()
    manifest = manifest_of(workspace)
    assert manifest.seed == 99
    assert manifest.metrics["weight_total"] == 50.0
    assert manifest.metrics["valid.lines"] == 28.0
    assert config_of(workspace)["seed"] == 99
    total = sum(
        item["weight"]
        for name, item in config_of(workspace)["data"].items()
        if name != "valid"
    )
    # The originals weigh 50 and the KD corpora as much: a `1:1` mix doubles the total.
    assert total == 100
    assert (workspace.out_dir / "data" / "bt" / "r0.5_s99").is_dir()
