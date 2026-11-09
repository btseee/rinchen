"""`rinchen export parity`: what it prints and writes (the line, the record `parity.json` and
the manifest), which rows it hands to the parity and with what, and its exit codes. What it
refuses and the order of its work are in `test_cli_export_parity_refusals.py`.

No test starts Eole or CTranslate2: the command calls `strict_parity` through its module-level
name, which `tests.support.parity_cli.FakeStrictParity` replaces. The workspace corpus has
three dev rows, two train rows and two test rows in each of the four toy directions, so the
sources of a run say which rows were read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.train.export_parity import ParityReport, ParitySource
from tests.support.fake_ct2 import SPIECE
from tests.support.fake_parity_eole import WEIGHTS_SHA256
from tests.support.parity_cli import (
    DEV_ROWS,
    EXPORT_NAME,
    MODEL_SHA256,
    FakeStrictParity,
    ParityWorkspace,
    install_parity,
    invoke,
    make_workspace,
    perfect_report,
)

DIRECTIONS = (
    "eng_Latn-khk_Cyrl",
    "khk_Cyrl-eng_Latn",
    "khk_Cyrl-rus_Cyrl",
    "rus_Cyrl-khk_Cyrl",
)
TARGETS = ("khk_Cyrl", "eng_Latn", "rus_Cyrl", "khk_Cyrl")


@pytest.fixture
def workspace(tmp_path: Path) -> ParityWorkspace:
    return make_workspace(tmp_path)


@pytest.fixture
def parity(monkeypatch: pytest.MonkeyPatch) -> FakeStrictParity:
    return install_parity(monkeypatch)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_of(workspace: ParityWorkspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def dev_sources(corpus: Path, direction: str, count: int) -> list[str]:
    """The source texts of the `count` dev rows of `direction` with the smallest ids."""
    src_lang, tgt_lang = direction.split("-")
    frame = pl.read_parquet(corpus)
    chosen = (
        frame.filter(
            (pl.col("split") == "dev")
            & (pl.col("src_lang") == src_lang)
            & (pl.col("tgt_lang") == tgt_lang)
        )
        .sort("id", "src", "tgt")
        .head(count)
    )
    return chosen["src"].to_list()


# --- help --------------------------------------------------------------------------------------


def test_help_names_every_option_and_marks_exactly_the_four_required_ones() -> None:
    result = invoke(["export", "parity", "--help"])

    assert result.exit_code == 0
    for option in (
        "--checkpoint",
        "--export",
        "--corpus",
        "--out-dir",
        "--max-lines",
        "--threads",
        "--timeout",
    ):
        assert option in result.stdout, option
    assert result.stdout.count("[required]") == 4


def test_export_help_lists_parity() -> None:
    assert "parity" in invoke(["export", "--help"]).stdout


# --- a passing run -----------------------------------------------------------------------------


def test_a_run_prints_the_parity_line_the_result_and_the_manifest(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    result = invoke(workspace.args())

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "parity: 12/12 exact, mean |delta| 8.30e-07 per token (gated) pass",
        f"result: {workspace.out_dir}/parity.json",
        f"manifest: {workspace.out_dir}/manifest.json",
    ]
    assert "error:" not in result.stderr


def test_the_result_file_is_the_report_and_what_names_the_inputs(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    invoke(workspace.args())

    assert workspace.result.read_text(encoding="utf-8") == (
        "{\n"
        f'  "checkpoint_weights_sha256": "{WEIGHTS_SHA256}",\n'
        '  "exact": 12,\n'
        '  "exact_share": 1.0,\n'
        f'  "export_name": "{EXPORT_NAME}",\n'
        '  "gated": true,\n'
        '  "lines": {\n'
        '    "eng_Latn-khk_Cyrl": 3,\n'
        '    "khk_Cyrl-eng_Latn": 3,\n'
        '    "khk_Cyrl-rus_Cyrl": 3,\n'
        '    "rus_Cyrl-khk_Cyrl": 3\n'
        "  },\n"
        '  "max_abs_delta": 3.5e-06,\n'
        '  "max_lines": 50,\n'
        '  "max_mean_delta": 0.001,\n'
        '  "mean_abs_delta": 8.3e-07,\n'
        '  "min_exact_share": 0.95,\n'
        '  "mismatches": [],\n'
        f'  "model_sha256": "{MODEL_SHA256}",\n'
        '  "passes": true,\n'
        '  "placement": "src",\n'
        '  "quantization": "float32",\n'
        '  "sentences": 12\n'
        "}\n"
    )


def test_a_non_ascii_mismatch_is_written_as_it_is(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = (f"{SPIECE}энэ", f"{SPIECE}тэр")
    install_parity(
        monkeypatch,
        report=lambda sources: perfect_report(
            sources, exact=11, exact_share=11 / 12, mismatches=(pair,)
        ),
    )

    invoke(workspace.args())

    text = workspace.result.read_bytes().decode("utf-8")
    assert pair[0] in text
    assert pair[1] in text


def test_the_same_inputs_give_the_same_result_bytes(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    invoke(workspace.args())
    first = workspace.result.read_bytes()

    invoke(workspace.args())

    assert workspace.result.read_bytes() == first


def test_the_manifest_vouches_for_the_inputs_and_the_result(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    invoke(workspace.args())

    manifest = manifest_of(workspace)
    export = workspace.export.directory
    assert manifest.stage == "export_parity"
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (str(workspace.corpus), sha256(workspace.corpus)),
        *(
            (str(export / name), sha256(export / name))
            for name in (
                "model.bin",
                "shared_vocabulary.json",
                "config.json",
                "spm.model",
                "rinchen.json",
                "manifest.json",
            )
        ),
        (str(workspace.weights), sha256(workspace.weights)),
        (
            str(workspace.checkpoint / "config.json"),
            sha256(workspace.checkpoint / "config.json"),
        ),
        (
            str(workspace.checkpoint / "vocab.json"),
            sha256(workspace.checkpoint / "vocab.json"),
        ),
    ]
    assert [(item.path, item.bytes) for item in manifest.inputs] == [
        (item.path, Path(item.path).stat().st_size) for item in manifest.inputs
    ]
    assert [(item.path, item.sha256) for item in manifest.outputs] == [
        (str(workspace.result), sha256(workspace.result))
    ]
    assert manifest.seed == -1
    assert manifest.status == {
        "parity": "pass",
        "quantization": "float32",
        "tag_placement": "src",
    }
    assert manifest.metrics == {
        "sentences": 12.0,
        "exact": 12.0,
        "exact_share": 1.0,
        "mean_abs_delta": 8.3e-07,
        "max_abs_delta": 3.5e-06,
        "gated": 1.0,
        "passes": 1.0,
        "lines.eng_Latn-khk_Cyrl": 3.0,
        "lines.khk_Cyrl-eng_Latn": 3.0,
        "lines.khk_Cyrl-rus_Cyrl": 3.0,
        "lines.rus_Cyrl-khk_Cyrl": 3.0,
    }
    assert set(manifest.tool_versions) == {
        "eole",
        "ctranslate2",
        "sentencepiece",
        "polars",
    }


def test_the_manifest_names_the_options_that_decide_the_numbers(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    invoke(workspace.args())
    default = manifest_of(workspace).config_sha256

    invoke(workspace.args("--max-lines", "2"))

    assert manifest_of(workspace).config_sha256 != default
    assert default == hashlib.sha256(b'{"max_lines":50}').hexdigest()


# --- what the parity is given -------------------------------------------------------------------


def test_the_parity_gets_the_dev_rows_of_every_direction_of_the_model(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    invoke(workspace.args())

    (call,) = parity.calls
    assert len(call.sources) == 4 * DEV_ROWS
    expected = [
        (text, target)
        for direction, target in zip(DIRECTIONS, TARGETS, strict=True)
        for text in dev_sources(workspace.corpus, direction, DEV_ROWS)
    ]
    assert [(s.text, s.tgt_lang) for s in call.sources] == expected
    assert all(" dev " in source.text for source in call.sources)


def test_max_lines_limits_each_direction(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    invoke(workspace.args("--max-lines", "2"))

    (call,) = parity.calls
    expected = [
        (text, target)
        for direction, target in zip(DIRECTIONS, TARGETS, strict=True)
        for text in dev_sources(workspace.corpus, direction, 2)
    ]
    assert [(s.text, s.tgt_lang) for s in call.sources] == expected
    assert '"max_lines": 2' in workspace.result.read_text(encoding="utf-8")
    assert json.loads(workspace.result.read_text(encoding="utf-8"))[
        "lines"
    ] == dict.fromkeys(DIRECTIONS, 2)
    assert manifest_of(workspace).metrics["lines.khk_Cyrl-rus_Cyrl"] == 2.0


def test_the_parity_gets_the_options_the_paths_and_the_digest_taken_before(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    invoke(workspace.args("--threads", "3", "--timeout", "90"))

    (call,) = parity.calls
    assert call.threads == 3
    assert call.timeout == 90.0
    assert call.checkpoint == workspace.checkpoint
    assert call.out_dir == workspace.out_dir
    assert call.export.directory == workspace.export.directory
    assert call.export.record.name == EXPORT_NAME
    assert call.weights_sha256 == WEIGHTS_SHA256


def test_the_defaults_are_two_threads_and_no_timeout(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    invoke(workspace.args())

    (call,) = parity.calls
    assert (call.threads, call.timeout) == (2, None)


def test_a_relative_checkpoint_path_reaches_the_parity_as_an_absolute_one(
    workspace: ParityWorkspace,
    parity: FakeStrictParity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(workspace.root)

    result = invoke(workspace.args(checkpoint=Path("ckpt"), out_dir=Path("out")))

    assert result.exit_code == 0, result.output
    (call,) = parity.calls
    assert call.checkpoint == workspace.checkpoint
    assert call.out_dir == workspace.out_dir


def test_a_run_creates_the_output_directory_and_replaces_an_earlier_result(
    workspace: ParityWorkspace, parity: FakeStrictParity
) -> None:
    workspace.out_dir.mkdir()
    workspace.result.write_text("old", encoding="utf-8")

    invoke(workspace.args())

    assert workspace.result.read_text(encoding="utf-8").startswith("{")


# --- a quantised export and a failing one --------------------------------------------------------


def test_a_quantised_export_says_it_is_not_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_workspace(tmp_path, quantization="int8")
    install_parity(
        monkeypatch,
        report=lambda sources: perfect_report(
            sources, quantization="int8", gated=False, mean_abs_delta=1.45e-03
        ),
    )

    result = invoke(workspace.args())

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == (
        "parity: 12/12 exact, mean |delta| 1.45e-03 per token (not gated: quantized) pass"
    )
    manifest = manifest_of(workspace)
    assert manifest.status["quantization"] == "int8"
    assert manifest.metrics["gated"] == 0.0


def failing(sources: list[ParitySource] | tuple[ParitySource, ...]) -> ParityReport:
    return perfect_report(
        sources,
        exact=11,
        exact_share=11 / 12,
        mismatches=(("a b", "a c"),),
    )


def test_a_run_that_does_not_pass_prints_fail_and_exits_4_with_everything_written(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_parity(monkeypatch, report=failing)

    result = invoke(workspace.args())

    assert result.exit_code == 4
    assert result.stdout.splitlines() == [
        "parity: 11/12 exact, mean |delta| 8.30e-07 per token (gated) FAIL",
        f"result: {workspace.out_dir}/parity.json",
        f"manifest: {workspace.out_dir}/manifest.json",
    ]
    document = json.loads(workspace.result.read_text(encoding="utf-8"))
    assert document["passes"] is False
    assert document["mismatches"] == [["a b", "a c"]]
    assert manifest_of(workspace).status["parity"] == "fail"
    assert manifest_of(workspace).metrics["passes"] == 0.0


def test_a_failing_mean_delta_of_a_gated_export_exits_4(
    workspace: ParityWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_parity(
        monkeypatch,
        report=lambda sources: perfect_report(sources, mean_abs_delta=1.5e-03),
    )

    result = invoke(workspace.args())

    assert result.exit_code == 4
    assert result.stdout.splitlines()[0].endswith("(gated) FAIL")
