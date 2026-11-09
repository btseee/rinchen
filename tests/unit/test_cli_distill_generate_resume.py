"""`rinchen distill generate`: how a run resumes (the shards are the work files), what a
crash and an interrupt leave, and how the command drives the REAL `TeacherTranslator` (over a
scripted CTranslate2: nothing loads a model) with the options it was given.

The shards of a run are `shards-<fingerprint>/<direction>/shard-<index>.parquet`; with
`--shard-size 2` the planted corpus (four English and three Russian sources) has two shards
in each direction, translated in the order English 0, English 1, Russian 0, Russian 1.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from rinchen.errors import UpstreamError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from tests.support.corpus_rows import EN_MN, RU_MN, corpus_row
from tests.support.distill_cli import (
    CliTeacher,
    GenerateWorkspace,
    install_ct2_teacher,
    install_teacher,
    invoke,
    make_generate_workspace,
    run_ok,
)
from tests.support.fake_ct2 import SPIECE, ScriptedCt2

TWO_SHARDS_EACH = ["--shard-size", "2"]
KD_PAIRS = {
    ("Hello", "Сайн байна уу"),
    ("Good morning", "Өглөөний мэнд"),
    ("Thank you", "Баярлалаа танд"),
    ("Water", "Хүйтэн ус"),
}
CT2_OPTIONS = {
    "beam_size": 4,
    "num_hypotheses": 1,
    "max_batch_size": 64,
    "max_decoding_length": 256,
    "return_end_token": True,
    "return_scores": False,
}


@pytest.fixture
def workspace(tmp_path: Path) -> GenerateWorkspace:
    return make_generate_workspace(tmp_path)


def manifest_of(workspace: GenerateWorkspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def kd_pairs(workspace: GenerateWorkspace) -> set[tuple[str, str]]:
    return {
        (row["src"], row["tgt"])
        for row in pl.read_parquet(workspace.kd_corpus).to_dicts()
    }


def shard_path(workspace: GenerateWorkspace, direction: str, index: int) -> Path:
    (directory,) = workspace.shard_directories()
    return directory / direction / f"shard-{index:05d}.parquet"


# --- resuming ----------------------------------------------------------------------------------


def test_a_rerun_reuses_every_shard_and_writes_the_manifest_again(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    run_ok(workspace.args(*TWO_SHARDS_EACH))
    first = manifest_of(workspace)
    seam = install_teacher(monkeypatch, CliTeacher({}))

    result = run_ok(workspace.args(*TWO_SHARDS_EACH))

    assert seam.teacher.calls == []
    assert result.stderr.splitlines() == [
        f"shard {EN_MN} 1/2 reused",
        f"shard {EN_MN} 2/2 reused",
        f"shard {RU_MN} 1/2 reused",
        f"shard {RU_MN} 2/2 reused",
    ]
    assert "shards: planned 4, reused 4, generated 0, regenerated_invalid 0" in (
        result.stdout.splitlines()
    )
    second = manifest_of(workspace)
    assert second.metrics["shards.reused"] == 4.0
    assert (
        second.metrics["count.total.selected"] == first.metrics["count.total.selected"]
    )
    assert second.created != first.created
    assert kd_pairs(workspace) == KD_PAIRS


def test_a_crash_keeps_the_finished_shards_and_writes_no_manifest(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(
        monkeypatch, CliTeacher(fail_on_call={2: UpstreamError("the teacher is down")})
    )

    result = invoke(workspace.args(*TWO_SHARDS_EACH))

    assert result.exit_code == 4, result.output
    errors = [line for line in result.stderr.splitlines() if line.startswith("error:")]
    assert errors == [
        f"error: direction {RU_MN}, shard 0 of 2 failed: the teacher is down"
    ]
    assert workspace.shard_files() == [
        f"{EN_MN}/shard-00000.parquet",
        f"{EN_MN}/shard-00001.parquet",
    ]
    assert not workspace.manifest.exists()
    assert not workspace.kd_corpus.exists()
    assert not workspace.counts.exists()


def test_the_rerun_after_a_crash_translates_only_what_is_missing(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(
        monkeypatch, CliTeacher(fail_on_call={2: UpstreamError("the teacher is down")})
    )
    invoke(workspace.args(*TWO_SHARDS_EACH))
    seam = install_teacher(monkeypatch)

    result = run_ok(workspace.args(*TWO_SHARDS_EACH))

    assert [(src, tgt) for _, src, tgt in seam.teacher.calls] == [
        ("rus_Cyrl", "khk_Cyrl")
    ] * 2
    assert "shards: planned 4, reused 2, generated 2, regenerated_invalid 0" in (
        result.stdout.splitlines()
    )
    assert kd_pairs(workspace) == KD_PAIRS
    assert manifest_of(workspace).status["generate"] == "ok"


def test_a_crash_removes_the_manifest_of_an_earlier_run_and_leaves_its_outputs(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    run_ok(workspace.args())
    corpus = workspace.kd_corpus.read_bytes()
    install_teacher(
        monkeypatch, CliTeacher(fail_on_call={0: UpstreamError("the teacher is down")})
    )

    result = invoke(workspace.args("--force"))

    assert result.exit_code == 4, result.output
    assert not workspace.manifest.exists()
    assert workspace.kd_corpus.read_bytes() == corpus


def test_an_interrupt_exits_130_and_keeps_the_finished_shards_and_writes_no_manifest(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch, CliTeacher(fail_on_call={1: KeyboardInterrupt()}))

    result = invoke(workspace.args(*TWO_SHARDS_EACH))

    assert result.exit_code == 130, result.output
    assert workspace.shard_files() == [f"{EN_MN}/shard-00000.parquet"]
    assert not workspace.manifest.exists()
    assert not workspace.kd_corpus.exists()


def test_an_interrupt_leaves_no_part_file_beside_the_shards(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch, CliTeacher(fail_on_call={1: KeyboardInterrupt()}))

    invoke(workspace.args(*TWO_SHARDS_EACH))

    assert [path for path in workspace.out_dir.rglob("*.part")] == []


def test_a_damaged_shard_is_generated_again_and_counted(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    run_ok(workspace.args(*TWO_SHARDS_EACH))
    shard_path(workspace, EN_MN, 1).write_bytes(b"not a parquet file")
    seam = install_teacher(monkeypatch)

    result = run_ok(workspace.args(*TWO_SHARDS_EACH))

    assert len(seam.teacher.calls) == 1
    assert "shards: planned 4, reused 3, generated 1, regenerated_invalid 1" in (
        result.stdout.splitlines()
    )
    assert kd_pairs(workspace) == KD_PAIRS


def test_force_generates_every_shard_again(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    run_ok(workspace.args(*TWO_SHARDS_EACH))
    seam = install_teacher(monkeypatch)

    result = run_ok(workspace.args(*TWO_SHARDS_EACH, "--force"))

    assert len(seam.teacher.calls) == 4
    assert "shards: planned 4, reused 0, generated 4, regenerated_invalid 0" in (
        result.stdout.splitlines()
    )


def test_another_seed_is_another_shard_directory_and_nothing_is_reused(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    run_ok(workspace.args("--seed", "1"))
    seam = install_teacher(monkeypatch)

    run_ok(workspace.args("--seed", "2"))

    assert len(workspace.shard_directories()) == 2
    assert len(seam.teacher.calls) == 2
    assert manifest_of(workspace).seed == 2


def test_the_batch_size_and_the_thread_count_do_not_change_the_shard_directory(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    run_ok(workspace.args("--batch-size", "8", "--threads", "1"))
    seam = install_teacher(monkeypatch, CliTeacher({}))

    run_ok(workspace.args("--batch-size", "16", "--threads", "2"))

    assert len(workspace.shard_directories()) == 1
    assert seam.teacher.calls == []


# --- a source the teacher must not be asked about ---------------------------------------------------


def test_a_source_without_a_letter_is_not_sent_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = make_generate_workspace(
        tmp_path, rows=[corpus_row("Hello", "Сайн"), corpus_row("...", "Цэг")]
    )
    seam = install_teacher(monkeypatch, CliTeacher({"Hello": ["Сайн уу"]}))

    result = run_ok(workspace.args())

    assert [texts for texts, _, _ in seam.teacher.calls] == [("Hello",)]
    lines = result.stdout.splitlines()
    assert "letterless sources: 1 (not sent to the teacher)" in lines
    assert manifest_of(workspace).metrics["letterless_sources.total"] == 1.0
    assert manifest_of(workspace).metrics["count.total.no_candidates"] == 1.0


# --- the real teacher over a scripted CTranslate2 ---------------------------------------------------


def test_the_real_teacher_translates_with_the_tag_in_the_source_and_the_default_options(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = install_ct2_teacher(monkeypatch)

    result = run_ok(workspace.args())

    assert kd_pairs(workspace) == KD_PAIRS
    assert len(library.calls) == 2
    for call in library.calls:
        assert {line[0] for line in call.source} == {"<2khk_Cyrl>"}
        assert call.target_prefix is None
        assert call.options == CT2_OPTIONS
    assert (
        "teacher counters (this process only): sources 7, source_too_long 0, "
        "empty_source 0, truncated 0, line_break 0, reserved_symbol 0"
    ) in result.stdout.splitlines()


def test_the_beam_size_the_hypotheses_the_batch_and_the_decoding_length_reach_the_library(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = install_ct2_teacher(monkeypatch)

    run_ok(
        workspace.args(
            "--beam-size",
            "2",
            "--n",
            "2",
            "--batch-size",
            "8",
            "--max-decoding-length",
            "32",
        )
    )

    assert [call.options for call in library.calls] == [
        {
            **CT2_OPTIONS,
            "beam_size": 2,
            "num_hypotheses": 2,
            "max_batch_size": 8,
            "max_decoding_length": 32,
        }
    ] * 2


def test_a_source_over_the_piece_limit_is_not_sent_and_the_counters_say_so(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    library: ScriptedCt2 = install_ct2_teacher(monkeypatch)

    result = run_ok(workspace.args("--max-source-pieces", "1"))

    lines = result.stdout.splitlines()
    assert (
        "teacher counters (this process only): sources 7, source_too_long 2, "
        "empty_source 0, truncated 0, line_break 0, reserved_symbol 0"
    ) in lines
    sent = {tuple(line) for call in library.calls for line in call.source}
    assert ("<2khk_Cyrl>", f"{SPIECE}Good", f"{SPIECE}morning") not in sent
    assert ("<2khk_Cyrl>", f"{SPIECE}Hello") in sent
    assert manifest_of(workspace).metrics["count.total.no_candidates"] == 3.0
    assert manifest_of(workspace).metrics["teacher.source_too_long"] == 2.0


def test_importing_the_distill_commands_does_not_load_ctranslate2() -> None:
    code = (
        "import sys; import rinchen.cli.distill; "
        "assert 'ctranslate2' not in sys.modules, 'ctranslate2 was imported'"
    )

    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 0, done.stderr
