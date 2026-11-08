"""The D7 STOP gate: a tiny ablation of two variants trains and is evaluated through the
real `rinchen train ablate` command, on the CPU, with the real Eole.

The matrix is the checked-in `configs/eole/ablation_tiny.yaml` (the direction tag on
the source and on the target, temperature 0.5, bt ratio 0.5, 1,000 steps of 2,048
tokens over the tiny base config) and the data are the toy corpora of
`tests/support/tiny_parallel.py`, written to a temporary directory with a toy
SentencePiece model. Each command is a subprocess of the `rinchen` binary beside the
interpreter, bounded by `SUBPROCESS_TIMEOUT_SECONDS`.

One module fixture runs the first ablation (about two minutes) and takes a snapshot
of what the tests below look at; two more fixtures, in order, rerun it (every variant
is reused) and rerun it after a checkpoint was truncated (that variant is trained
again). The toy sentences are templated and the model saturates them, so the chrF
threshold checks the plumbing (the right tags, the right language for every
direction), not the quality a real corpus would show.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.train.eole_cfg import read_config, validate_train_config
from tests.support.tiny_parallel import TOY_DIRECTIONS
from tests.support.tiny_variant import ToyInputs, build_toy_inputs

# The tiny matrix runs two variants of 1,000 steps; a hung Eole must fail the test.
SUBPROCESS_TIMEOUT_SECONDS = 900
# The gate. The spike's worst direction over four seeds was 98; it is never lowered.
MIN_CHRF = 60.0
MATRIX = REPO_ROOT / "configs" / "eole" / "ablation_tiny.yaml"
RINCHEN = str(Path(sys.executable).with_name("rinchen"))
SRC_VARIANT = "t0.5_bt0.5_src_s1234"
TGT_VARIANT = "t0.5_bt0.5_tgt_s1234"
VARIANTS = (SRC_VARIANT, TGT_VARIANT)
DIRECTIONS = sorted(TOY_DIRECTIONS)
CHECKPOINT = "model.00.safetensors"
# A rerun that trains nothing: the training of one variant takes about a minute.
MAX_RERUN_SECONDS = 30.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Invocation:
    """One `rinchen train ablate` command: what it printed and how long it took."""

    returncode: int
    stdout: list[str]
    stderr: str
    seconds: float


def _ablate(
    inputs: ToyInputs, out_dir: Path, *extra: str, matrix: Path = MATRIX
) -> Invocation:
    started = time.perf_counter()
    completed = subprocess.run(
        [
            RINCHEN,
            "train",
            "ablate",
            "--matrix",
            str(matrix),
            "--corpus",
            str(inputs.corpus),
            "--spm-model",
            str(inputs.spm_model),
            "--out-dir",
            str(out_dir),
            "--threads",
            "2",
            "--timeout",
            str(SUBPROCESS_TIMEOUT_SECONDS),
            *extra,
        ],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )
    return Invocation(
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8").splitlines(),
        stderr=completed.stderr.decode("utf-8", "replace"),
        seconds=time.perf_counter() - started,
    )


def _ok(invocation: Invocation) -> Invocation:
    assert invocation.returncode == 0, invocation.stderr
    return invocation


def _tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _outcomes(invocation: Invocation) -> list[tuple[str, str]]:
    """(variant id, trained or reused) of each progress line."""
    return [
        (words[2], words[3])
        for words in (line.split() for line in invocation.stdout)
        if words[0] == "variant" and "/" in words[1]
    ]


@dataclass(frozen=True)
class FirstRun:
    """The first ablation and what was on disk right after it."""

    inputs: ToyInputs
    out_dir: Path
    invocation: Invocation
    tsv: list[dict[str, str]]
    results: dict[str, Any]
    manifest: Manifest
    output_sha256: dict[str, str]
    result_json: dict[str, dict[str, Any]]
    training_logs_ns: dict[str, int]


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> FirstRun:
    root = tmp_path_factory.mktemp("train_ablate")
    inputs = build_toy_inputs(root / "in")
    out_dir = root / "out"
    invocation = _ok(_ablate(inputs, out_dir))
    manifest = store.read(out_dir)
    assert manifest is not None
    return FirstRun(
        inputs=inputs,
        out_dir=out_dir,
        invocation=invocation,
        tsv=_tsv_rows(out_dir / "results.tsv"),
        results=json.loads((out_dir / "results.json").read_text(encoding="utf-8")),
        manifest=manifest,
        output_sha256={
            item.path: _sha256(Path(item.path)) for item in manifest.outputs
        },
        result_json={
            variant: json.loads(
                (out_dir / "variants" / variant / "result.json").read_text("utf-8")
            )
            for variant in VARIANTS
        },
        training_logs_ns={
            variant: (out_dir / "variants" / variant / "run" / "train.log")
            .stat()
            .st_mtime_ns
            for variant in VARIANTS
        },
    )


@pytest.fixture(scope="module")
def rerun(first_run: FirstRun) -> Invocation:
    return _ok(_ablate(first_run.inputs, first_run.out_dir))


@pytest.fixture(scope="module")
def rerun_after_truncation(first_run: FirstRun, rerun: Invocation) -> Invocation:
    """The run after the checkpoint of the source-tag variant was cut short."""
    checkpoint = (
        first_run.out_dir / "variants" / SRC_VARIANT / "run" / "ckpt" / CHECKPOINT
    )
    with checkpoint.open("r+b") as handle:
        handle.truncate(checkpoint.stat().st_size // 2)
    return _ok(_ablate(first_run.inputs, first_run.out_dir))


@pytest.mark.subprocess
def test_the_first_run_trains_both_variants_and_names_the_result_files(
    first_run: FirstRun, record_property: Any
) -> None:
    record_property("seconds", round(first_run.invocation.seconds, 1))
    lines = first_run.invocation.stdout

    assert _outcomes(first_run.invocation) == [
        (SRC_VARIANT, "trained"),
        (TGT_VARIANT, "trained"),
    ]
    assert "variants: 2 (trained 2, evaluated 0, reused 0)" in lines
    assert [line.split(": ")[0] for line in lines[-3:]] == [
        "results.tsv",
        "results.json",
        "results.md",
    ]
    for line in lines[-3:]:
        assert Path(line.split(": ", 1)[1]).is_file()


@pytest.mark.subprocess
def test_results_tsv_has_a_header_and_two_rows_of_four_directions(
    first_run: FirstRun,
) -> None:
    rows = first_run.tsv

    assert [row["variant"] for row in rows] == list(VARIANTS)
    for column in ("variant", "chrf_mean", "chrf_min", "tag_leaks", "train_seconds"):
        assert column in rows[0]
    for direction in DIRECTIONS:
        assert f"chrf_{direction}" in rows[0]
        assert f"bleu_{direction}" in rows[0]
    assert len([name for name in rows[0] if name.startswith("chrf_")]) == 4 + 2


@pytest.mark.subprocess
def test_every_direction_of_both_variants_reaches_the_chrf_gate(
    first_run: FirstRun, record_property: Any
) -> None:
    for row in first_run.tsv:
        scores = {
            direction: (
                float(row[f"chrf_{direction}"]),
                float(row[f"bleu_{direction}"]),
            )
            for direction in DIRECTIONS
        }
        record_property(f"scores {row['variant']}", scores)
        assert float(row["chrf_min"]) >= MIN_CHRF, (row["variant"], scores)
        for direction, (chrf, _bleu) in scores.items():
            assert chrf >= MIN_CHRF, (row["variant"], direction, chrf)
    for variant in first_run.results["variants"]:
        assert variant["dev"]["chrf_min"] >= MIN_CHRF
        assert variant["dev"]["tag_leaks"] == 0
        assert sorted(variant["dev"]["directions"]) == DIRECTIONS


@pytest.mark.subprocess
def test_the_rendered_configs_pass_the_validator_and_their_weights_sum_to_100(
    first_run: FirstRun,
) -> None:
    for variant in first_run.results["variants"]:
        cfg = read_config(
            first_run.out_dir / "variants" / variant["id"] / "config.yaml", "config"
        )
        validate_train_config(cfg)
        weights = {
            name: corpus["weight"]
            for name, corpus in cfg["data"].items()
            if name != "valid"
        }
        assert sum(weights.values()) == 100
        assert weights == {item["name"]: item["weight"] for item in variant["corpora"]}
        assert variant["tag_placement"] in ("src", "tgt")


@pytest.mark.subprocess
def test_the_validation_corpus_of_the_checkpoint_carries_no_prefix(
    first_run: FirstRun,
) -> None:
    for variant in VARIANTS:
        checkpoint_dir = first_run.out_dir / "variants" / variant / "run" / "ckpt"
        saved = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))

        valid = saved["data"]["valid"]
        assert not {"src_prefix", "tgt_prefix", "src_suffix", "tgt_suffix"} & set(valid)
        assert not valid.get("transforms")


@pytest.mark.subprocess
def test_the_manifest_hashes_match_the_files_and_the_results(
    first_run: FirstRun,
) -> None:
    manifest = first_run.manifest

    assert manifest.stage == "train_ablate"
    assert manifest.status == {"ablation": "ok"}
    assert manifest.metrics["variants.total"] == 2.0
    assert manifest.metrics["variants.trained"] == 2.0
    assert manifest.metrics["threads"] == 2.0
    outputs = {item.path: item.sha256 for item in manifest.outputs}
    for variant in VARIANTS:
        checkpoint = str(
            first_run.out_dir / "variants" / variant / "run" / "ckpt" / CHECKPOINT
        )
        assert outputs[checkpoint] == first_run.output_sha256[checkpoint]
    for variant in VARIANTS:
        checkpoint = (
            first_run.out_dir / "variants" / variant / "run" / "ckpt" / CHECKPOINT
        )
        assert (
            outputs[str(checkpoint)]
            == first_run.result_json[variant]["checkpoint_sha256"]
        )
        assert str(first_run.out_dir / "variants" / variant / "config.yaml") in outputs
    assert [item.sha256 for item in manifest.inputs] == [
        _sha256(path)
        for path in (
            MATRIX,
            REPO_ROOT / "configs" / "eole" / "multilingual_tiny.yaml",
            first_run.inputs.corpus,
            first_run.inputs.spm_model,
        )
    ]
    assert list(first_run.out_dir.rglob("*.part")) == []


@pytest.mark.subprocess
def test_a_dry_run_into_a_fresh_directory_creates_nothing(
    first_run: FirstRun, tmp_path: Path
) -> None:
    fresh = tmp_path / "fresh"

    dry = _ok(_ablate(first_run.inputs, fresh, "--dry-run"))

    assert not fresh.exists()
    assert dry.stdout[-1] == "nothing was trained and nothing was written"
    assert "variants: 2" in dry.stdout
    assert any(line.startswith(f"variant {SRC_VARIANT} corpus ") for line in dry.stdout)


@pytest.mark.subprocess
def test_the_second_run_reuses_both_variants_within_seconds(
    first_run: FirstRun, rerun: Invocation, record_property: Any
) -> None:
    record_property("rerun_seconds", round(rerun.seconds, 1))

    assert _outcomes(rerun) == [(SRC_VARIANT, "reused"), (TGT_VARIANT, "reused")]
    assert "variants: 2 (trained 0, evaluated 0, reused 2)" in rerun.stdout
    assert rerun.seconds < MAX_RERUN_SECONDS
    for variant in VARIANTS:
        log = first_run.out_dir / "variants" / variant / "run" / "train.log"
        assert log.stat().st_mtime_ns == first_run.training_logs_ns[variant]
    assert _tsv_rows(first_run.out_dir / "results.tsv") == first_run.tsv


@pytest.mark.subprocess
def test_a_truncated_checkpoint_is_trained_again_and_the_other_variant_is_not(
    first_run: FirstRun, rerun: Invocation, rerun_after_truncation: Invocation
) -> None:
    assert _outcomes(rerun_after_truncation) == [
        (SRC_VARIANT, "trained"),
        (TGT_VARIANT, "reused"),
    ]
    tgt_log = first_run.out_dir / "variants" / TGT_VARIANT / "run" / "train.log"
    assert tgt_log.stat().st_mtime_ns == first_run.training_logs_ns[TGT_VARIANT]
    src_log = first_run.out_dir / "variants" / SRC_VARIANT / "run" / "train.log"
    assert src_log.stat().st_mtime_ns > first_run.training_logs_ns[SRC_VARIANT]
    manifest = store.read(first_run.out_dir)
    assert manifest is not None
    outputs = {item.path: item.sha256 for item in manifest.outputs}
    for variant in VARIANTS:
        checkpoint = (
            first_run.out_dir / "variants" / variant / "run" / "ckpt" / CHECKPOINT
        )
        assert outputs[str(checkpoint)] == _sha256(checkpoint)
        # Same seed and thread count: the second training reproduces the first one's
        # checkpoint byte for byte (spec finding 8).
        assert outputs[str(checkpoint)] == first_run.output_sha256[str(checkpoint)]
    rows = _tsv_rows(first_run.out_dir / "results.tsv")
    assert all(float(row["chrf_min"]) >= MIN_CHRF for row in rows)
