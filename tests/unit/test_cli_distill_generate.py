"""`rinchen distill generate`: what it plans and prints, what it writes (the KD corpus, the
counts and the manifest) and how the options reach the teacher. How it resumes and how the
real teacher is driven is in `test_cli_distill_generate_resume.py`, the QE filter in
`test_cli_distill_generate_qe.py` and what it refuses in
`test_cli_distill_generate_refusals.py`.

The teacher is a scripted fake behind the command's `build_teacher` seam
(`tests.support.distill_cli`); the corpus is `tests.support.planted_kd`, whose seven sources
and what a teacher with `PLANTED_SCRIPT` makes of them are worked out there: four English
sources keep a target and the three Russian ones keep none (a refusal, a Latin answer, a
copy). Nothing starts CTranslate2.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from rinchen.cli import train_common
from rinchen.config.synth import SYNTH_CONFIG_PATH
from rinchen.corpus.schema import make_row_id, validate_corpus_schema
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.train.distill_export import verify_export
from rinchen.train.distill_generate import kd_fingerprint
from rinchen.train.distill_teacher import TeacherOptions
from tests.support.corpus_rows import EN_MN, RU_MN
from tests.support.distill_cli import (
    DEFAULT_NAME,
    CliTeacher,
    GenerateWorkspace,
    install_scorer,
    install_teacher,
    invoke,
    make_generate_workspace,
    run_ok,
    write_selection_config,
)
from tests.support.planted_kd import planted_sources
from tests.support.step_dirs import sha256

TEACHER_COUNTERS = {
    "sources": 7,
    "source_too_long": 0,
    "empty_source": 0,
    "truncated": 2,
    "line_break": 1,
}
COUNT_LINE = (
    "sources {sources}, candidates {candidates}, dropped_empty {dropped_empty}, "
    "dropped_purity {dropped_purity}, dropped_qe {dropped_qe}, dropped_copy "
    "{dropped_copy}, no_candidates {no_candidates}, no_survivor {no_survivor}, "
    "selected {selected}"
)
EN_MN_COUNTS = {
    "sources": 4,
    "candidates": 4,
    "dropped_empty": 0,
    "dropped_purity": 0,
    "dropped_qe": 0,
    "dropped_copy": 0,
    "no_candidates": 0,
    "no_survivor": 0,
    "selected": 4,
}
RU_MN_COUNTS = {
    "sources": 3,
    "candidates": 2,
    "dropped_empty": 0,
    "dropped_purity": 1,
    "dropped_qe": 0,
    "dropped_copy": 1,
    "no_candidates": 1,
    "no_survivor": 2,
    "selected": 0,
}
TOTAL_COUNTS = {
    "sources": 7,
    "candidates": 6,
    "dropped_empty": 0,
    "dropped_purity": 1,
    "dropped_qe": 0,
    "dropped_copy": 1,
    "no_candidates": 1,
    "no_survivor": 2,
    "selected": 4,
}
KD_PAIRS = {
    ("Hello", "Сайн байна уу"),
    ("Good morning", "Өглөөний мэнд"),
    ("Thank you", "Баярлалаа танд"),
    ("Water", "Хүйтэн ус"),
}
SHARD_FILES_OF_ONE_SHARD = [
    f"{EN_MN}/shard-00000.parquet",
    f"{RU_MN}/shard-00000.parquet",
]
SHARD_PROGRESS = [
    f"shard {EN_MN} 1/1 generated",
    f"shard {RU_MN} 1/1 generated",
]
DECODING_LINE = (
    "decoding: beam size 4, n 1, max decoding length 256, max source pieces 256, "
    "batch size 64, device cpu, compute type default, threads 1"
)


@pytest.fixture
def workspace(tmp_path: Path) -> GenerateWorkspace:
    return make_generate_workspace(tmp_path)


def manifest_of(workspace: GenerateWorkspace) -> Manifest:
    manifest = store.read(workspace.out_dir)
    assert manifest is not None
    return manifest


def fingerprint_of(
    workspace: GenerateWorkspace,
    *,
    shard_size: int = 512,
    seed: int = 1234,
    options: TeacherOptions | None = None,
) -> str:
    return kd_fingerprint(
        verify_export(workspace.teacher),
        options or TeacherOptions(),
        planted_sources(),
        shard_size=shard_size,
        seed=seed,
    )


def plan_lines(
    workspace: GenerateWorkspace,
    *,
    name: str = DEFAULT_NAME,
    shards: tuple[int, int] = (1, 1),
    reusable: int = 0,
    decoding: str = DECODING_LINE,
    shard_size: int = 512,
    seed: int = 1234,
    qe: str = "not requested",
) -> list[str]:
    """The plan a run prints before it translates, for the planted corpus."""
    fingerprint = fingerprint_of(workspace, shard_size=shard_size, seed=seed)
    return [
        "corpus files: 4",
        f"teacher: {name} (tag placement src)",
        f"direction {EN_MN}: sources 4, shards {shards[0]}",
        f"direction {RU_MN}: sources 3, shards {shards[1]}",
        "sources: 7",
        f"shards: {sum(shards)} (reusable {reusable})",
        decoding,
        f"qe: {qe}",
        f"fingerprint: {fingerprint}",
        f"shard directory: {workspace.out_dir}/shards-{fingerprint[:12]}",
    ]


def result_lines(workspace: GenerateWorkspace) -> list[str]:
    """What a run of the planted corpus with `PLANTED_SCRIPT` and the counters of
    `TEACHER_COUNTERS` prints after it translated."""
    return [
        f"counts {EN_MN}: {COUNT_LINE.format(**EN_MN_COUNTS)}",
        f"counts {RU_MN}: {COUNT_LINE.format(**RU_MN_COUNTS)}",
        f"counts total: {COUNT_LINE.format(**TOTAL_COUNTS)}",
        "matches reference: 3 of 4",
        "letterless sources: 0 (not sent to the teacher)",
        (
            "teacher counters (this process only): sources 7, source_too_long 0, "
            "empty_source 0, truncated 2, line_break 1"
        ),
        "shards: planned 2, reused 0, generated 2, regenerated_invalid 0",
        "qe: skipped: not requested",
        f"kd corpus: {workspace.kd_corpus}",
        f"manifest: {workspace.manifest}",
    ]


# --- help ----------------------------------------------------------------------------------


def test_help_names_every_option_and_marks_exactly_the_three_required_ones() -> None:
    result = invoke(["distill", "generate", "--help"])

    assert result.exit_code == 0
    for option in (
        "--corpus",
        "--teacher",
        "--out-dir",
        "--direction",
        "--beam-size",
        "--n",
        "--max-decoding-length",
        "--max-source-pieces",
        "--batch-size",
        "--shard-size",
        "--device",
        "--compute-type",
        "--threads",
        "--seed",
        "--name",
        "--selection-config",
        "--qe",
        "--force",
        "--dry-run",
    ):
        assert option in result.stdout, option
    assert result.stdout.count("[required]") == 3
    assert "<cpu|cuda>" in result.stdout


def test_the_help_states_the_defaults() -> None:
    text = " ".join(invoke(["distill", "generate", "--help"]).stdout.split())

    for option, default in (
        ("--beam-size", "4"),
        ("--n", "1"),
        ("--max-decoding-length", "256"),
        ("--max-source-pieces", "256"),
        ("--batch-size", "64"),
        ("--shard-size", "512"),
        ("--device", "cpu"),
        ("--compute-type", "default"),
        ("--threads", "1"),
        ("--seed", "1234"),
    ):
        assert (
            f"[default: {default}]" in text.split(f"{option} ", 1)[1].split(" --", 1)[0]
        ), option
    assert "[default:" in text.split("--selection-config ", 1)[1].split(" --", 1)[0]


# --- the dry run -----------------------------------------------------------------------------


def test_a_dry_run_prints_the_plan_and_says_nothing_was_written(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seam = install_teacher(monkeypatch)

    result = run_ok(workspace.args("--dry-run", "--name", "toy"))

    assert result.stdout.splitlines() == [
        *plan_lines(workspace, name="toy"),
        "nothing was written",
    ]
    assert seam.built == []


def test_a_dry_run_creates_nothing_not_even_the_output_directory(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args("--dry-run"))

    assert not workspace.out_dir.exists()


def test_a_dry_run_writes_nothing_beside_an_output_directory_that_exists(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    workspace.out_dir.mkdir()

    run_ok(workspace.args("--dry-run"))

    assert list(workspace.out_dir.iterdir()) == []


def test_a_dry_run_counts_the_shards_a_smaller_shard_size_makes(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    result = run_ok(workspace.args("--dry-run", "--shard-size", "2"))

    assert result.stdout.splitlines() == [
        *plan_lines(workspace, shards=(2, 2), shard_size=2),
        "nothing was written",
    ]


def test_a_dry_run_says_how_many_shards_a_run_would_reuse(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    run_ok(workspace.args("--shard-size", "2"))

    result = run_ok(workspace.args("--dry-run", "--shard-size", "2"))

    assert result.stdout.splitlines() == [
        *plan_lines(workspace, shards=(2, 2), reusable=4, shard_size=2),
        "nothing was written",
    ]


def test_a_dry_run_with_force_says_no_shard_would_be_reused(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    run_ok(workspace.args("--shard-size", "2"))

    result = run_ok(workspace.args("--dry-run", "--shard-size", "2", "--force"))

    assert result.stdout.splitlines() == [
        *plan_lines(workspace, shards=(2, 2), reusable=0, shard_size=2),
        "nothing was written",
    ]


def test_a_dry_run_and_the_run_that_follows_it_agree_on_the_plan(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    dry = run_ok(workspace.args("--dry-run")).stdout.splitlines()

    run = run_ok(workspace.args()).stdout.splitlines()

    assert run[: len(dry) - 1] == dry[:-1]


def test_the_qe_line_of_the_plan_says_whether_the_filter_was_asked_for(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    seam = install_scorer(monkeypatch, None)

    result = run_ok(workspace.args("--dry-run", "--qe"))

    assert "qe: requested" in result.stdout.splitlines()
    assert seam.built == []


# --- what a run prints --------------------------------------------------------------------------


def test_a_run_prints_the_plan_then_the_counts_and_where_it_wrote(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch, CliTeacher(counters=TEACHER_COUNTERS))

    result = run_ok(workspace.args())

    assert result.stdout.splitlines() == [
        *plan_lines(workspace),
        *result_lines(workspace),
    ]


def test_the_progress_of_each_shard_goes_to_standard_error(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    result = run_ok(workspace.args())

    assert result.stderr.splitlines() == SHARD_PROGRESS


def test_the_teacher_is_built_once_from_the_verified_export_and_the_options(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seam = install_teacher(monkeypatch)

    run_ok(workspace.args())

    ((export, options),) = seam.built
    assert export.directory == workspace.teacher
    assert dict(export.sha256) == workspace.export.sha256
    assert export.manifest_sha256 == workspace.export.manifest_sha256
    assert options == TeacherOptions()


def test_every_option_of_the_teacher_reaches_it(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The seam is faked: `cuda` names a device and nothing here runs on one.
    seam = install_teacher(monkeypatch)

    run_ok(
        workspace.args(
            "--beam-size",
            "6",
            "--n",
            "3",
            "--max-decoding-length",
            "99",
            "--max-source-pieces",
            "77",
            "--batch-size",
            "8",
            "--device",
            "cuda",
            "--compute-type",
            "int8",
            "--threads",
            "3",
        )
    )

    ((_, options),) = seam.built
    assert options == TeacherOptions(
        beam_size=6,
        n=3,
        max_decoding_length=99,
        max_source_pieces=77,
        batch_size=8,
        device="cuda",
        compute_type="int8",
        threads=3,
    )


def test_the_directions_are_checked_once_before_the_first_translation(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seam = install_teacher(monkeypatch)

    run_ok(workspace.args())

    assert seam.teacher.checked == [((EN_MN, RU_MN), 0)]
    assert len(seam.teacher.calls) == 2


# --- what a run writes -------------------------------------------------------------------------


def test_a_run_writes_the_corpus_the_counts_the_shards_and_the_manifest_only(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args())

    assert sorted(
        name for name in workspace.written() if not name.startswith("shards-")
    ) == ["counts.json", "kd.parquet", "manifest.json"]
    assert workspace.shard_files() == SHARD_FILES_OF_ONE_SHARD
    (directory,) = workspace.shard_directories()
    assert directory.name == f"shards-{fingerprint_of(workspace)[:12]}"


def test_the_kd_corpus_is_in_the_corpus_schema_and_holds_the_teachers_best_targets(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args("--name", "toy"))

    frame = pl.read_parquet(workspace.kd_corpus)
    validate_corpus_schema(frame.schema, source="kd.parquet")
    assert {(row["src"], row["tgt"]) for row in frame.to_dicts()} == KD_PAIRS
    assert frame.height == 4
    assert set(frame["origin"].cast(pl.String)) == {"ft"}
    assert set(frame["corpus"]) == {"ft:toy"}
    assert set(frame["split"].cast(pl.String)) == {"train"}
    assert set(frame["src_lang"]) == {"eng_Latn"}
    assert set(frame["tgt_lang"]) == {"khk_Cyrl"}
    assert frame["filter_reason"].null_count() == 4
    assert frame["doc_id"].null_count() == 4
    for row in frame.to_dicts():
        assert row["id"] == make_row_id("eng_Latn", "khk_Cyrl", row["src"], row["tgt"])


def test_the_default_name_of_the_corpus_is_made_from_the_hash_of_the_model(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args())

    assert set(pl.read_parquet(workspace.kd_corpus)["corpus"]) == {f"ft:{DEFAULT_NAME}"}
    assert manifest_of(workspace).status["teacher.name"] == DEFAULT_NAME


def test_counts_json_holds_the_run_the_teacher_the_counts_and_the_statistics(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch, CliTeacher(counters=TEACHER_COUNTERS))

    run_ok(workspace.args("--name", "toy", "--seed", "5"))

    document = workspace.counts_document()
    assert sorted(document) == [
        "counts",
        "decoding",
        "fingerprint",
        "letterless_sources",
        "matches_reference",
        "name",
        "qe_status",
        "seed",
        "selection",
        "shard_size",
        "shard_stats",
        "teacher",
        "teacher_counters",
        "teacher_counters_cover",
    ]
    assert document["fingerprint"] == fingerprint_of(workspace, seed=5)
    assert document["name"] == "toy"
    assert document["seed"] == 5
    assert document["shard_size"] == 512
    assert document["counts"] == {
        EN_MN: EN_MN_COUNTS,
        RU_MN: RU_MN_COUNTS,
        "total": TOTAL_COUNTS,
    }
    assert document["matches_reference"] == {EN_MN: 3, RU_MN: 0, "total": 3}
    assert document["letterless_sources"] == {EN_MN: 0, RU_MN: 0, "total": 0}
    assert document["shard_stats"] == {
        "planned": 2,
        "reused": 0,
        "generated": 2,
        "regenerated_invalid": 0,
    }
    assert document["teacher_counters"] == TEACHER_COUNTERS
    assert "this process" in document["teacher_counters_cover"]
    assert document["qe_status"] == "skipped: not requested"
    assert document["decoding"] == asdict(TeacherOptions())
    assert document["selection"]["min_script_share"] == 0.9
    assert document["selection"]["mbr_utility"] == "chrf"


def test_counts_json_names_the_teacher_by_its_hashes_and_its_checkpoint(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args("--name", "toy"))

    teacher = workspace.counts_document()["teacher"]
    assert teacher["name"] == "toy"
    assert teacher["tag_placement"] == "src"
    assert teacher["sha256"] == workspace.export.sha256
    assert teacher["manifest_sha256"] == workspace.export.manifest_sha256
    assert teacher["checkpoint"] == asdict(workspace.export.record.checkpoint)


# --- the manifest --------------------------------------------------------------------------------


def test_the_manifest_vouches_for_every_input_with_the_digest_it_had_before_it_was_read(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args())

    export = workspace.export
    manifest = manifest_of(workspace)
    assert [(Path(item.path), item.sha256) for item in manifest.inputs] == [
        *((path, sha256(path)) for path in workspace.corpora),
        (SYNTH_CONFIG_PATH, sha256(SYNTH_CONFIG_PATH)),
        *(
            (workspace.teacher / name, export.sha256[name])
            for name in (
                "model.bin",
                "shared_vocabulary.json",
                "config.json",
                "spm.model",
                "export.json",
            )
        ),
        (workspace.teacher / "manifest.json", export.manifest_sha256),
    ]
    assert manifest.config_sha256 == sha256(SYNTH_CONFIG_PATH)


def test_the_manifest_lists_the_corpus_with_its_rows_and_the_counts_file(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args())

    manifest = manifest_of(workspace)
    assert [(Path(item.path), item.rows) for item in manifest.outputs] == [
        (workspace.kd_corpus, 4),
        (workspace.counts, None),
    ]
    for item in manifest.outputs:
        assert item.sha256 == sha256(Path(item.path)), item.path
        assert item.bytes == Path(item.path).stat().st_size, item.path


def test_the_manifest_is_the_stage_the_prepare_command_reads(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args())

    manifest = manifest_of(workspace)
    assert manifest.stage == "distill_generate"
    assert manifest.seed == 1234
    assert manifest.status["generate"] == "ok"


def test_the_metrics_hold_the_counts_by_direction_and_in_total(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch, CliTeacher(counters=TEACHER_COUNTERS))

    run_ok(workspace.args())

    metrics = manifest_of(workspace).metrics
    assert {
        key.removeprefix("count.total."): value
        for key, value in metrics.items()
        if key.startswith("count.total.")
    } == {key: float(value) for key, value in TOTAL_COUNTS.items()}
    assert metrics[f"count.{EN_MN}.selected"] == 4.0
    assert metrics[f"count.{RU_MN}.dropped_purity"] == 1.0
    assert metrics[f"count.{RU_MN}.dropped_copy"] == 1.0
    assert metrics[f"count.{RU_MN}.no_survivor"] == 2.0


def test_the_metrics_hold_the_statistics_the_decoding_and_the_teachers_counters(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch, CliTeacher(counters=TEACHER_COUNTERS))

    run_ok(workspace.args())

    metrics = manifest_of(workspace).metrics
    assert metrics["matches_reference.total"] == 3.0
    assert metrics[f"matches_reference.{EN_MN}"] == 3.0
    assert metrics[f"matches_reference.{RU_MN}"] == 0.0
    assert metrics["letterless_sources.total"] == 0.0
    assert {
        key: value for key, value in metrics.items() if key.startswith("shards.")
    } == {
        "shards.planned": 2.0,
        "shards.reused": 0.0,
        "shards.generated": 2.0,
        "shards.regenerated_invalid": 0.0,
    }
    assert {
        key: value for key, value in metrics.items() if key.startswith("teacher.")
    } == {f"teacher.{key}": float(value) for key, value in TEACHER_COUNTERS.items()}
    assert {
        key: value for key, value in metrics.items() if key.startswith("decoding.")
    } == {
        "decoding.beam_size": 4.0,
        "decoding.n": 1.0,
        "decoding.max_decoding_length": 256.0,
        "decoding.max_source_pieces": 256.0,
        "decoding.batch_size": 64.0,
        "decoding.threads": 1.0,
    }
    assert metrics["shard_size"] == 512.0


def test_the_status_names_the_teacher_by_its_checkpoint_and_says_how_it_decoded(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    run_ok(workspace.args("--name", "toy"))

    assert manifest_of(workspace).status == {
        "generate": "ok",
        "qe": "skipped: not requested",
        "teacher.name": "toy",
        "teacher.tag_placement": "src",
        "teacher.checkpoint_weights_sha256": "1" * 64,
        "teacher.checkpoint_manifest_sha256": "4" * 64,
        "teacher.model_sha256": workspace.export.sha256["model.bin"],
        "decoding": (
            '{"batch_size":64,"beam_size":4,"compute_type":"default","device":"cpu",'
            '"max_decoding_length":256,"max_source_pieces":256,"n":1,"threads":1}'
        ),
    }


def test_the_manifest_records_the_installed_tools(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata

    install_teacher(monkeypatch)

    run_ok(workspace.args())

    assert manifest_of(workspace).tool_versions == {
        package: importlib.metadata.version(package)
        for package in (
            "ctranslate2",
            "sentencepiece",
            "polars",
            "sacrebleu",
            "eole",
            "torch",
        )
    }


def test_the_manifest_is_written_after_every_file_it_lists(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)
    seen: list[tuple[list[bool], bool]] = []
    real_write = train_common.store.write

    def looking(out_dir: Path, manifest: Manifest) -> Path:
        seen.append(
            (
                [Path(item.path).is_file() for item in manifest.outputs],
                (out_dir / "manifest.json").exists(),
            )
        )
        return real_write(out_dir, manifest)

    monkeypatch.setattr(train_common.store, "write", looking)

    run_ok(workspace.args())

    assert seen == [([True, True], False)]


# --- the selection -------------------------------------------------------------------------------


def test_the_selection_config_changes_what_is_kept(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 18 of the 19 letters are Cyrillic: pure enough for the default 0.9, not for 1.0.
    script: dict[str, Any] = {
        "Water": ["Хүйтэн ус хэрэглэнэ ч x"],
        "Hello": ["Сайн байна уу"],
        "Good morning": ["Өглөөний мэнд"],
        "Thank you": ["Баярлалаа танд"],
        "Привет": ["Сайн уу"],
        "Спасибо": ["Баярлалаа"],
        "Книга": ["Ном"],
    }
    install_teacher(monkeypatch, CliTeacher(script))
    default = run_ok(workspace.args("--name", "a"))
    strict_config = write_selection_config(
        tmp_path / "strict.yaml", min_script_share=1.0
    )

    strict = run_ok(
        workspace.args(
            "--name",
            "a",
            "--selection-config",
            str(strict_config),
            out_dir=tmp_path / "strict-out",
        )
    )

    assert (
        f"counts {EN_MN}: " + COUNT_LINE.format(**{**EN_MN_COUNTS, "selected": 4})
        in default.stdout.splitlines()
    )
    assert (
        f"counts {EN_MN}: "
        + COUNT_LINE.format(
            **{**EN_MN_COUNTS, "dropped_purity": 1, "no_survivor": 1, "selected": 3}
        )
        in strict.stdout.splitlines()
    )
    manifest = store.read(tmp_path / "strict-out")
    assert manifest is not None
    assert manifest.config_sha256 == sha256(strict_config)
    assert Path(manifest.inputs[4].path) == strict_config
    document = json.loads((tmp_path / "strict-out" / "counts.json").read_text("utf-8"))
    assert document["selection"]["min_script_share"] == 1.0


# --- the direction option and an empty selection ---------------------------------------------------


def test_the_direction_option_restricts_the_sources_the_checks_and_the_counts(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seam = install_teacher(monkeypatch)

    result = run_ok(workspace.args("--direction", EN_MN))

    lines = result.stdout.splitlines()
    assert f"direction {EN_MN}: sources 4, shards 1" in lines
    assert not any(line.startswith(f"direction {RU_MN}") for line in lines)
    assert not any(line.startswith(f"counts {RU_MN}") for line in lines)
    assert seam.teacher.checked == [((EN_MN,), 0)]
    assert [(src, tgt) for _, src, tgt in seam.teacher.calls] == [
        ("eng_Latn", "khk_Cyrl")
    ]
    assert workspace.shard_files() == [f"{EN_MN}/shard-00000.parquet"]
    assert list(workspace.counts_document()["counts"]) == [EN_MN, "total"]


def test_a_direction_given_twice_is_one_direction(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seam = install_teacher(monkeypatch)

    run_ok(workspace.args("--direction", EN_MN, "--direction", EN_MN))

    assert seam.teacher.checked == [((EN_MN,), 0)]


def test_a_selection_that_keeps_nothing_writes_an_empty_corpus_and_says_so(
    workspace: GenerateWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_teacher(monkeypatch)

    result = run_ok(workspace.args("--direction", RU_MN))

    assert pl.read_parquet(workspace.kd_corpus).height == 0
    assert manifest_of(workspace).outputs[0].rows == 0
    assert (
        f"counts total: {COUNT_LINE.format(**RU_MN_COUNTS)}"
        in result.stdout.splitlines()
    )
