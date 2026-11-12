"""The D15 STOP gate, the recipe and what it wrote (R1 to R3, R5, R6 and R13): the body of the `tiny` recipe
of the justfile, run through `sh` in a temporary directory, through the real `rinchen` binary and the real
Eole, CTranslate2, SentencePiece and sacrebleu on the CPU with every device hidden. The controls (R7 to R12)
are in `test_tiny_gate_controls.py`, the determinism (R4) in `test_tiny_gate_determinism.py` and the hygiene
(R14) in `test_tiny_gate_hygiene.py`.

The `uv` the recipe finds is a shim (`tests/support/fake_uv.py`): `run` goes to the real uv as `run
--project <repository> --no-sync --frozen --offline` and everything else is refused, and the shim's call log
is the proof that nothing else was asked of uv. The gate cannot prove the absence of a socket in a child
process, and it does not use `no_network` (the guard of one process): what keeps the run off the network is the
shim, the offline switches and the empty device list of every child, and the loopback listener of the synthetic
data stage is the one socket the pipeline opens. No GPU was used (the one on this machine was not authorised).

Gates (every floor a literal of `tests/support/tiny_gate.py` and never lowered):

    R1  the recipe exits 0 inside 1,200 s and writes a report of schema `rinchen.tiny/1`, verdict `pass`, the
        fifteen stage ids in order, every stage `ok`; the parts that are skipped are `filter.lid` (no GlotLID
        model) and the token-ratio note of `filter.length`, nothing else; the eight thresholds `pass`
        (`train_chrf` at least 50.0, `parity_exact` at least 0.95, `parity_delta` at most 1e-3,
        `int8_chrf_drop` at most 1.0, `script_purity` 1.0, `placeholder_acc` 1.0, `wall_time` at most 1,200
        and the `stages` threshold); the `synth` block shows every row of the monolingual fixture read as a
        source, every source selected and one row for each, and the `train` block shows back-translated rows in
        the mix (`synth_problems`: no stage and no threshold reads them, and a server that answers nothing
        still lets the synthetic command exit 0); the shim's log holds
        one `run` call and no refusal
    R2  `report.md` is the rendering of `report.json` and both documents hold no absolute path, date or time,
        duration key, value of the environment, secret by its shape or text of a fixture
    R3  the manifest: stage `tiny_run`, written after both documents, the 13 inputs with the digests of their
        bytes, every name of `uv.lock` and `python` in `tool_versions`, the seconds of all fifteen stages, the
        seven outputs listed with their digests
    R5  the filter: 770 rows, 756 kept, the ten planted rows caught with exactly their reasons, `filter.lid`
        skipped for its closed reason, no `holdout` row in `train.parquet`
    R6  one flipped byte of a fixture: exit 3 in under 5 s with nothing written
    R13 the service stage: the statuses `ok`, the counter 1.0 and the refusal 400 in the report's `serve` block

Measured on this machine (CPU, 20 cores, shared with other work; Eole 0.6.4, CTranslate2 4.8.4, uv 0.13.7,
2026-11-12; a time moves by a third between runs, a count, a score or a digest never does):

    R1  the recipe ran in 113.1 s when it was alone (load average 1.2, 2.4 and 4.1 when it started, 2.5, 2.4 and
           3.9 when it ended) and in 121.0 s beside one other pipeline; the manifest's seconds in the first run: environment 0.05, fetch 0.00, corpus 0.03, filter 0.43, spm 1.83,
           converter 41.24, synth 1.05, prepare 0.97, train 44.16, average 0.92, export 6.37, parity 3.52,
           serve 0.06, robustness 1.28, evaluate 10.27 (112.2 s in all); `train_chrf` 99.3585 (the least of
           five directions, `mn_Mong-eng_Latn`), `parity_exact` 1.0 (70 of 70), `parity_delta` 1.62e-06,
           `int8_chrf_drop` 0.0, `script_purity` 1.0, `placeholder_acc` 1.0 (238 of 238); the shim logged
           one call
    R2  report.json 26,545 bytes, report.md (its size is not pinned: the gate compares it with the rendering
           of report.json and between runs), manifest.json 11,513 to 11,564 bytes (it holds the seconds and a
           creation time: it differs from run to run)
    R3  13 inputs, 187 `tool_versions` keys (186 names and `python`), 7 outputs
    R5  `reasons` duplicate 4, labse 3, len_ratio_char 3, numbers 2, script_tgt 2; 4 of 760 authentic rows
           flagged; train.parquet 1,316 rows (1,246 train, 70 dev)

UNMEASURED: GitHub Actions and a 4-core runner (the recipe was run only here), the real state of the package
index, anything on a GPU and every real model. The toy ranks nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest

from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import STAGE_IDS
from rinchen.tiny.report import parse_report
from rinchen.tiny.report_markdown import render
from rinchen.tiny.states import PART_IDS, SKIP_NO_LID, SKIP_TOKEN_RATIO
from tests.support.fake_uv import read_calls
from tests.support.maintain_gate import lock_table
from tests.support.tiny_gate import (
    JUSTFILE,
    MAX_INT8_CHRF_DROP,
    MAX_PARITY_DELTA,
    MAX_WALL_TIME_SECONDS,
    MIN_PARITY_EXACT,
    MIN_PLACEHOLDER_ACC,
    MIN_SCRIPT_PURITY,
    MIN_TRAIN_CHRF,
    RecipeRun,
    absolute_paths,
    duration_keys,
    fixture_mono_rows,
    fixture_text_hits,
    fixture_texts,
    leaked_values,
    part_states,
    read_manifest,
    recipe_command,
    run_tiny,
    secret_like_strings,
    stage_of,
    synth_problems,
    threshold_of,
    timestamps,
)
from tests.support.tiny_profiles import control_profile, copy_fixtures

pytestmark = [pytest.mark.subprocess, pytest.mark.usefixtures("tiny_gpu_hidden")]

FIXTURES: Final = REPO_ROOT / "tests" / "fixtures" / "parallel"
FIXTURE_FILES: Final = (
    "eng_Latn-khk_Cyrl.tsv",
    "rus_Cyrl-khk_Cyrl.tsv",
    "mn_Mong.tsv",
    "mono_khk_Cyrl.tsv",
    "noise.tsv",
    "nouns.tsv",
    "entities.tsv",
)
# The inputs the manifest hashes before anything reads them, relative to the repository, in order.
MANIFEST_INPUTS: Final = (
    "configs/tiny/tiny.yaml",
    *(f"tests/fixtures/parallel/{name}" for name in FIXTURE_FILES),
    "configs/eole/tiny.yaml",
    "configs/serve/tiny.yaml",
    "tests/fixtures/script_conv/tiny_words.tsv",
    "configs/profiles/tiny.yaml",
    "uv.lock",
)
# The outputs, relative to the output directory, after the two documents.
CHECKPOINT_OUTPUTS: Final = (
    "04_spm/tiny.model",
    "05_converter/model/model.bin",
    "07_train/ablate/variants/t1_bt0.2_src_s1234/run/averaged/model.00.safetensors",
    "08_export/float32/model.bin",
    "08_export/int8/model.bin",
)
THE_COMMAND: Final = [
    "run",
    "--frozen",
    "--no-sync",
    "rinchen",
    "tiny",
    "run",
    "--profile",
    "tiny",
]
# The one skip the filter has (the lid), with its reason from the closed list; every other part ran.
FILTER_STATES: Final = {
    "script_ratio": "ok",
    "numbers": "ok",
    "length": "ok",
    "lid": "skipped",
    "minhash": "ok",
    "labse": "ok",
    "qe": "ok",
}
REJECTION_REASONS: Final = {
    "duplicate": 4,
    "labse": 3,
    "len_ratio_char": 3,
    "numbers": 2,
    "script_tgt": 2,
}
PLANTED_REASONS: Final = {
    "duplicate": 2,
    "labse": 2,
    "len_ratio_char": 2,
    "numbers": 2,
    "script_tgt": 2,
}
SERVICE_STATUS_CODES: Final = [200, 200, 200, 400, 200]
PIPELINE_ENTRIES: Final = (
    "01_fetch",
    "02_corpus",
    "03_filter",
    "04_spm",
    "05_converter",
    "06_synth",
    "07_train",
    "08_export",
    "09_parity",
    "10_robustness",
    "11_evaluate",
    "manifest.json",
    "report.json",
    "report.md",
)
REFUSAL_SECONDS: Final = 5.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =================================================================================
# R1: the recipe, natively.


def test_the_recipe_that_ran_is_the_body_of_the_justfile(
    tiny_run_dir: RecipeRun,
) -> None:
    command = recipe_command(JUSTFILE)

    assert command == "uv run --frozen --no-sync rinchen tiny run --profile tiny"
    assert read_calls(tiny_run_dir.shim_dir)[0]["argv"] == THE_COMMAND


def test_the_recipe_exits_zero_inside_the_budget_and_writes_the_documents(
    tiny_run_dir: RecipeRun, record_property: Any
) -> None:
    invocation = tiny_run_dir.invocation
    record_property("recipe_seconds", round(invocation.seconds, 1))
    record_property("load_start", tiny_run_dir.load_start)
    record_property("load_end", tiny_run_dir.load_end)

    assert invocation.returncode == 0, invocation.stderr[-3000:]
    assert invocation.seconds <= MAX_WALL_TIME_SECONDS
    assert "verdict: pass" in invocation.stdout
    assert {path.name for path in tiny_run_dir.out_dir.iterdir()} == set(
        PIPELINE_ENTRIES
    )


def test_the_report_passes_and_lists_the_fifteen_stages_in_order_all_ok(
    tiny_report: dict[str, Any],
) -> None:
    assert tiny_report["schema"] == "rinchen.tiny/1"
    assert tiny_report["profile"] == "tiny"
    assert tiny_report["qe_fake"] is True
    assert tiny_report["verdict"] == "pass"
    assert [stage["id"] for stage in tiny_report["stages"]] == list(STAGE_IDS)
    assert len(STAGE_IDS) == 15
    for stage in tiny_report["stages"]:
        assert (stage["state"], stage["reason"]) == ("ok", None), stage["id"]


def test_the_synthetic_run_selected_every_source_and_the_training_mix_holds_its_rows(
    tiny_report: dict[str, Any], record_property: Any
) -> None:
    assert synth_problems(tiny_report, fixture_mono_rows(FIXTURES)) == []

    synth = tiny_report["blocks"]["synth"]
    record_property("synth_counts", synth["counts"])
    record_property("synth_rows", synth["rows"])
    record_property("bt_rows", tiny_report["blocks"]["train"]["bt_rows"])


def test_only_the_lid_and_the_token_ratio_check_are_skipped_and_each_says_why(
    tiny_report: dict[str, Any],
) -> None:
    filtered = stage_of(tiny_report, "filter")
    evaluated = stage_of(tiny_report, "evaluate")

    assert list(PART_IDS["filter"]) == list(FILTER_STATES)
    assert part_states(filtered) == FILTER_STATES
    (lid,) = [part for part in filtered["parts"] if part["id"] == "lid"]
    assert lid["reason"] == SKIP_NO_LID
    assert filtered["detail"]["token_ratio_check"] == SKIP_TOKEN_RATIO
    assert part_states(evaluated) == {
        "memorised": "ok",
        "heldout": "ok",
        "neural_seam": "ok",
        "report": "ok",
    }
    assert all(
        part["reason"] is None
        for part in [*filtered["parts"], *evaluated["parts"]]
        if part["state"] == "ok"
    )
    skipped = [
        (stage["id"], part["id"])
        for stage in tiny_report["stages"]
        for part in stage["parts"]
        if part["state"] == "skipped"
    ]
    assert skipped == [("filter", "lid")]


def test_the_eight_thresholds_pass_with_the_floors_of_the_master_plan(
    tiny_report: dict[str, Any], record_property: Any
) -> None:
    thresholds = tiny_report["thresholds"]
    assert [item["id"] for item in thresholds] == [
        "stages",
        "train_chrf",
        "parity_exact",
        "parity_delta",
        "int8_chrf_drop",
        "script_purity",
        "placeholder_acc",
        "wall_time",
    ]
    for item in thresholds:
        record_property(f"threshold_{item['id']}", item["value"])
        assert item["verdict"] == "pass", item["id"]

    assert threshold_of(tiny_report, "stages")["value"] == 0
    train = threshold_of(tiny_report, "train_chrf")
    assert (train["bound"], train["value"] >= MIN_TRAIN_CHRF) == (
        f">= {MIN_TRAIN_CHRF!r}",
        True,
    )
    exact = threshold_of(tiny_report, "parity_exact")
    assert exact["bound"] == f">= {MIN_PARITY_EXACT!r}"
    assert exact["value"] >= MIN_PARITY_EXACT
    delta = threshold_of(tiny_report, "parity_delta")
    assert delta["bound"] == f"<= {MAX_PARITY_DELTA!r}"
    assert delta["value"] <= MAX_PARITY_DELTA
    drop = threshold_of(tiny_report, "int8_chrf_drop")
    assert drop["bound"] == f"<= {MAX_INT8_CHRF_DROP!r}"
    assert drop["value"] <= MAX_INT8_CHRF_DROP
    purity = threshold_of(tiny_report, "script_purity")
    assert purity["bound"] == f">= {MIN_SCRIPT_PURITY!r}"
    assert purity["value"] >= MIN_SCRIPT_PURITY
    placeholders = threshold_of(tiny_report, "placeholder_acc")
    assert placeholders["bound"] == f">= {MIN_PLACEHOLDER_ACC!r}"
    assert placeholders["value"] >= MIN_PLACEHOLDER_ACC
    wall = threshold_of(tiny_report, "wall_time")
    assert wall["bound"] == f"<= {MAX_WALL_TIME_SECONDS!r}"


def test_the_train_chrf_is_the_least_memorised_figure_of_the_int8_export(
    tiny_report: dict[str, Any],
) -> None:
    """The threshold is read over the five memorised directions of the product (the int8 export), at the
    training rows, and over no other set: it is recomputed here from the blocks of the report."""
    rows = [
        row for row in tiny_report["blocks"]["memorised"] if row["system"] == "int8"
    ]
    held_out = [row for row in tiny_report["blocks"]["heldout"]]

    assert sorted(row["direction"] for row in rows) == [
        "eng_Latn-khk_Cyrl",
        "khk_Cyrl-eng_Latn",
        "khk_Cyrl-rus_Cyrl",
        "mn_Mong-eng_Latn",
        "rus_Cyrl-khk_Cyrl",
    ]
    assert threshold_of(tiny_report, "train_chrf")["value"] == min(
        row["chrf"] for row in rows
    )
    assert held_out, "the held-out rows are reported and never gated"
    assert min(row["chrf"] for row in held_out) < min(row["chrf"] for row in rows)


def test_every_decode_behind_a_threshold_was_greedy(tiny_run_dir: RecipeRun) -> None:
    """The decode of every thresholded run is beam 1 (a placeholder is lost at beam 4 in one export of
    nine): the options each run recorded say so, for the robustness measurement and for every run of
    the evaluation that decoded with the model."""
    out_dir = tiny_run_dir.out_dir
    robustness = json.loads(
        (out_dir / "10_robustness" / "robustness.json").read_text(encoding="utf-8")
    )
    runs = sorted((out_dir / "11_evaluate" / "runs").glob("*/*/run.json"))
    decoded = [
        document["system"]
        for document in (json.loads(path.read_text(encoding="utf-8")) for path in runs)
        if document["system"]["kind"] == "rinchen"
    ]

    assert robustness["system"]["options"]["beam_size"] == 1
    # Three memorised suites in two exports and two held-out suites in one: eight decodes.
    assert len(decoded) == 8
    assert {system["options"]["beam_size"] for system in decoded} == {1}


def test_the_shim_was_asked_for_one_run_and_refused_nothing(
    tiny_run_dir: RecipeRun,
) -> None:
    calls = read_calls(tiny_run_dir.shim_dir)

    assert len(calls) == 1
    assert calls[0]["forwarded"] is True
    assert calls[0]["refused"] is False
    assert calls[0]["final"][1:] == [
        "run",
        "--project",
        str(REPO_ROOT),
        "--no-sync",
        "--frozen",
        "--offline",
        "rinchen",
        "tiny",
        "run",
        "--profile",
        "tiny",
    ]


# =================================================================================
# R2: the documents.


def test_the_markdown_is_the_rendering_of_the_json_alone(
    tiny_run_dir: RecipeRun,
) -> None:
    out_dir = tiny_run_dir.out_dir
    data = (out_dir / "report.json").read_bytes()

    assert (out_dir / "report.md").read_text(encoding="utf-8") == render(
        parse_report(data)
    )


@pytest.mark.parametrize("name", ["report.json", "report.md"])
def test_neither_document_holds_a_path_a_date_a_secret_or_a_text_of_a_fixture(
    tiny_run_dir: RecipeRun, name: str, record_property: Any
) -> None:
    path = tiny_run_dir.out_dir / name
    text = path.read_text(encoding="utf-8")
    environment = {
        variable: value
        for variable in ("HOME", "USER", "LOGNAME", "PATH", "PWD", "SHELL", "TMPDIR")
        if (value := os.environ.get(variable))
    }
    environment["hostname"] = platform.node()
    environment["working directory"] = str(tiny_run_dir.cwd)
    texts = fixture_texts(FIXTURES)
    record_property("bytes", path.stat().st_size)

    assert len(texts) > 1000  # the scan is not vacuous
    assert absolute_paths(text) == []
    assert timestamps(text) == []
    assert secret_like_strings(text) == []
    assert fixture_text_hits(text, texts) == 0
    assert leaked_values(text, environment) == []


def test_the_report_holds_no_duration_the_wall_time_has_a_verdict_and_no_figure(
    tiny_report: dict[str, Any],
) -> None:
    """The verdict of the wall time is the figure's and the seconds are the manifest's: no key of the
    report names a duration or a date, and the threshold of the wall time carries no value."""
    assert duration_keys(tiny_report) == []
    assert threshold_of(tiny_report, "wall_time")["value"] is None


# =================================================================================
# R3: the manifest.


def test_the_manifest_is_the_runs_and_was_written_after_both_documents(
    tiny_run_dir: RecipeRun,
) -> None:
    out_dir = tiny_run_dir.out_dir
    manifest = read_manifest(out_dir)
    stamps = {
        name: (out_dir / name).stat().st_mtime_ns
        for name in ("report.json", "report.md", "manifest.json")
    }

    assert manifest["stage"] == "tiny_run"
    assert manifest["seed"] == 1234
    assert re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?", manifest["code_version"])
    assert stamps["manifest.json"] >= max(stamps["report.json"], stamps["report.md"])
    assert manifest["status"] == {stage: "ok" for stage in STAGE_IDS}


def test_the_inputs_are_the_thirteen_files_hashed_before_they_were_read(
    tiny_run_dir: RecipeRun,
) -> None:
    manifest = read_manifest(tiny_run_dir.out_dir)
    named = [
        (Path(item["path"]).relative_to(REPO_ROOT).as_posix(), item)
        for item in manifest["inputs"]
    ]

    assert [name for name, _ in named] == list(MANIFEST_INPUTS)
    for name, item in named:
        assert item["sha256"] == _sha256(REPO_ROOT / name), name
        assert item["bytes"] == (REPO_ROOT / name).stat().st_size, name
    assert manifest["config_sha256"] == _sha256(REPO_ROOT / "configs/tiny/tiny.yaml")


def test_every_package_of_the_lock_and_the_interpreter_are_recorded(
    tiny_run_dir: RecipeRun,
) -> None:
    versions = read_manifest(tiny_run_dir.out_dir)["tool_versions"]
    locked = lock_table(REPO_ROOT / "uv.lock").versions

    assert len(locked) == 186
    assert {name: value for name, value in versions.items() if name != "python"} == {
        name: ", ".join(found) for name, found in locked.items()
    }
    assert versions["python"] == platform.python_version()


def test_the_seconds_of_each_stage_and_the_counts_are_in_the_manifest(
    tiny_run_dir: RecipeRun, tiny_report: dict[str, Any]
) -> None:
    metrics = read_manifest(tiny_run_dir.out_dir)["metrics"]

    assert [name for name in metrics if name.startswith("seconds.")] == [
        f"seconds.{stage}" for stage in STAGE_IDS
    ]
    assert all(metrics[f"seconds.{stage}"] >= 0.0 for stage in STAGE_IDS)
    assert sum(metrics[f"seconds.{stage}"] for stage in STAGE_IDS) <= (
        MAX_WALL_TIME_SECONDS
    )
    assert metrics["rows.kept"] == tiny_report["blocks"]["filter"]["kept"]
    assert metrics["train.steps"] == tiny_report["blocks"]["train"]["steps"]


def test_the_outputs_are_the_two_documents_and_the_five_checkpoint_files_by_digest(
    tiny_run_dir: RecipeRun,
) -> None:
    out_dir = tiny_run_dir.out_dir
    manifest = read_manifest(out_dir)
    listed = [
        (Path(item["path"]).relative_to(out_dir).as_posix(), item)
        for item in manifest["outputs"]
    ]

    assert [name for name, _ in listed] == [
        "report.json",
        "report.md",
        *CHECKPOINT_OUTPUTS,
    ]
    for name, item in listed:
        assert item["sha256"] == _sha256(out_dir / name), name


# =================================================================================
# R5: the filter.


def test_the_planted_rows_get_exactly_their_reasons_and_the_counts_are_the_fixtures(
    tiny_report: dict[str, Any],
) -> None:
    block = tiny_report["blocks"]["filter"]

    assert (block["rows"], block["kept"], block["train_rows"]) == (770, 756, 1316)
    assert block["authentic_rows"] == 760
    assert block["authentic_flagged"] == 4
    assert block["authentic_flagged_share"] <= 0.01
    assert block["reasons"] == REJECTION_REASONS
    planted = block["planted"]
    assert (planted["total"], planted["detectable"], planted["matched"]) == (10, 10, 10)
    assert planted["expected"] == planted["found"] == PLANTED_REASONS


def test_no_row_the_filter_held_out_reaches_the_training_file(
    tiny_run_dir: RecipeRun,
) -> None:
    directory = tiny_run_dir.out_dir / "03_filter"
    scored = pl.read_parquet(directory / "scored.parquet", glob=False)
    kept = pl.read_parquet(directory / "kept.parquet", glob=False)
    train = pl.read_parquet(directory / "train.parquet", glob=False)

    held_out = scored.filter(pl.col("split").cast(pl.String) == "holdout")
    assert held_out.height == 10
    assert held_out["filter_reason"].null_count() == 0
    assert kept.filter(pl.col("split").cast(pl.String) == "holdout").height == 0
    assert set(train["split"].cast(pl.String).to_list()) == {"train", "dev"}
    assert set(train["corpus"].cast(pl.String).to_list()).isdisjoint({"tiny-noise"})
    assert train.height == 1316
    # The ids are digests of the texts: the two planted duplicates of a training row share theirs with it
    # (by design), so the rows that were held out are told apart by their corpus and their split.


# =================================================================================
# R6: a changed fixture is refused before anything is written.


def test_one_flipped_byte_of_a_fixture_is_refused_with_exit_three_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    fixtures = copy_fixtures(tmp_path / "fixtures")
    changed = bytearray((fixtures / "eng_Latn-khk_Cyrl.tsv").read_bytes())
    changed[len(changed) // 2] ^= 0x01
    (fixtures / "eng_Latn-khk_Cyrl.tsv").write_bytes(bytes(changed))
    profile = control_profile(tmp_path, **{"fixtures.directory": str(fixtures)})
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    out_dir = tmp_path / "out"

    invocation = run_tiny(
        ["--config", profile, "--out-dir", out_dir], cwd=cwd, timeout=60
    )

    assert invocation.returncode == 3, invocation.stderr
    assert invocation.seconds < REFUSAL_SECONDS
    assert invocation.stdout == []
    assert "eng_Latn-khk_Cyrl.tsv" in invocation.stderr
    assert not out_dir.exists()
    assert list(cwd.iterdir()) == []


# =================================================================================
# R13: the service stage, inside the run.


def test_the_service_answered_every_request_and_refused_the_bad_one(
    tiny_report: dict[str, Any],
) -> None:
    serve = tiny_report["blocks"]["serve"]

    assert stage_of(tiny_report, "serve")["state"] == "ok"
    assert serve["status_codes"] == SERVICE_STATUS_CODES
    assert serve["statuses"] == ["ok", "ok"]
    assert serve["script_purity"] == [1.0, 1.0]
    assert serve["translate_200_total"] == 1.0
    assert serve["error_code"] == "bad_request"
