"""`launch(..., continue_from=...)`: a NEW run that starts from one of our own checkpoints.

Eole never starts: `run_logged` is replaced by `tests.support.fake_launch.FakeEole`, which
writes the logs and step directories a run leaves (it starts counting at step 1 when the
config says `reset_optim: all`, as Eole does). The sources are fabricated runs and averaged
models with real manifests (`tests.support.step_dirs`); what makes a source acceptable has
its tests in `test_train_continuation.py`. Here is what the launcher does with it: the
order of its checks, what it writes, which config each attempt gets and what a rerun,
an interrupted run, a finished run or another source makes of it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.train import launch as launch_module
from rinchen.train.checkpoints import (
    identity_of,
    read_launch_record,
    step_directories,
)
from rinchen.train.continuation import SourceProvenance, verify_source
from rinchen.train.launch import LaunchResult, launch
from rinchen.train.run_lock import holding_the_run
from tests.support import step_dirs
from tests.support.fake_launch import (
    EOLE,
    FakeEole,
    Script,
    continuation_config,
    write_config,
)
from tests.support.step_dirs import (
    SOURCE_VOCABULARY,
    FabricatedRun,
    make_averaged_dir,
    make_finished_run,
    sha256,
    tree_digest,
)

pytestmark = pytest.mark.usefixtures("spm_model_in_the_cwd")

FAILED = "eole train failed: exit code 1; log: x; last 20 lines:\nboom"


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def teacher(tmp_path: Path) -> FabricatedRun:
    return make_finished_run(tmp_path / "teacher")


@pytest.fixture
def base_cfg(run_dir: Path) -> dict[str, Any]:
    return continuation_config(run_dir)


@pytest.fixture
def config_path(tmp_path: Path, base_cfg: dict[str, Any]) -> Path:
    return write_config(tmp_path / "base.yaml", base_cfg)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, run_dir: Path) -> FakeEole:
    stand_in = FakeEole(run_dir, saved_vocabulary=SOURCE_VOCABULARY)
    monkeypatch.setattr(launch_module, "run_logged", stand_in)
    return stand_in


def go(
    config_path: Path, run_dir: Path, source: Path | None, **options: Any
) -> LaunchResult:
    return launch(config_path, run_dir, eole=EOLE, continue_from=source, **options)


def crash(step: int) -> Script:
    """A run killed after it reached `step` (it saved nothing there)."""
    return Script(to_step=step, skip_final_save=True, error=UpstreamError(FAILED))


def start_and_crash(
    fake: FakeEole,
    config_path: Path,
    run_dir: Path,
    source: Path,
    step: int = 120,
) -> None:
    fake.queue(crash(step))
    with pytest.raises(UpstreamError):
        go(config_path, run_dir, source)


def rendered(run_dir: Path, attempt: int = 1) -> dict[str, Any]:
    loaded = yaml.safe_load(
        (run_dir / "launch" / f"train-{attempt}.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


def files_under(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def record_document(run_dir: Path) -> dict[str, Any]:
    document = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def config_with(base: dict[str, Any], **training: Any) -> dict[str, Any]:
    changed = json.loads(json.dumps(base))
    changed["training"].update(training)
    return changed


# ---- the first attempt ------------------------------------------------------------------


def test_the_first_attempt_is_the_base_config_continuing_the_source_with_a_fresh_optimizer(
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    before = config_path.read_bytes()

    result = go(config_path, run_dir, teacher.final)

    expected = json.loads(json.dumps(base_cfg))
    expected["training"]["train_from"] = str(teacher.final)
    expected["training"]["reset_optim"] = "all"
    assert rendered(run_dir) == expected
    assert fake.trains[0].config == expected
    assert config_path.read_bytes() == before
    assert (result.status, result.steps, result.resumed_from, result.attempt) == (
        "completed",
        400,
        None,
        1,
    )


def test_the_vocabulary_is_built_from_the_attempt_config_then_topped_up_then_eole_trains(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    go(config_path, run_dir, teacher.final)

    assert [call.subcommand for call in fake.calls] == ["build_vocab", "train"]
    build_vocab, train = fake.calls
    assert build_vocab.config == train.config
    reserved = train.vocabulary or b""
    assert b"<doc>\t1\n" in reserved
    assert b"<sep>\t1\n" in reserved


def test_the_result_names_the_source_that_was_verified(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    result = go(config_path, run_dir, teacher.final)

    assert result.continued_from == verify_source(teacher.final)
    assert isinstance(result.continued_from, SourceProvenance)


def test_an_averaged_model_is_continued_the_same_way(
    tmp_path: Path, config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    averaged = make_averaged_dir(tmp_path / "averaged")

    result = go(config_path, run_dir, averaged)

    assert rendered(run_dir)["training"]["train_from"] == str(averaged)
    assert rendered(run_dir)["training"]["reset_optim"] == "all"
    assert result.continued_from is not None
    assert result.continued_from.kind == "averaged"
    assert record_document(run_dir)["continued_from"]["kind"] == "averaged"


def test_a_relative_source_is_given_to_eole_as_an_absolute_path(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    teacher: FabricatedRun,
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = go(config_path, run_dir, Path("teacher/run/ckpt/step_200"))

    assert rendered(run_dir)["training"]["train_from"] == str(teacher.final)
    assert result.continued_from is not None
    assert result.continued_from.path == teacher.final


def test_the_source_is_never_written(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    before = tree_digest(teacher.run_dir)
    start_and_crash(fake, config_path, run_dir, teacher.final)

    go(config_path, run_dir, teacher.final)

    assert tree_digest(teacher.run_dir) == before


def test_the_run_ends_with_the_steps_of_its_own_count_from_one(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    result = go(config_path, run_dir, teacher.final)

    assert result.checkpoint == run_dir / "ckpt" / "step_400"
    log = (run_dir / "launch" / "attempt-1.log").read_text(encoding="utf-8")
    assert "Step 10/  400;" in log
    assert {path.name for path in (run_dir / "ckpt").glob("step_*")} == {
        f"step_{step}" for step in range(50, 401, 50)
    }


# ---- the record -----------------------------------------------------------------------------


def test_the_record_is_version_2_and_holds_the_source(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    go(config_path, run_dir, teacher.final)

    document = record_document(run_dir)
    assert document["schema_version"] == 2
    assert document["continued_from"] == {
        "kind": "step",
        "path": str(teacher.final),
        "weights_sha256": sha256(teacher.final / "model.00.safetensors"),
        "optimizer_sha256": sha256(teacher.final / "optimizer.pt"),
        "config_sha256": sha256(teacher.final / "config.json"),
        "vocab_sha256": sha256(teacher.final / "vocab.json"),
        "manifest_sha256": sha256(teacher.manifest),
        "manifest_stage": "train_launch",
        "step": 200,
        "source_run": str(teacher.run_dir),
    }


def test_the_identity_of_the_run_includes_the_source(
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    go(config_path, run_dir, teacher.final)

    record = read_launch_record(run_dir)
    assert record is not None
    assert record.continued_from is not None
    assert record.identity == identity_of(
        base_cfg, continued_from=record.continued_from
    )
    assert record.identity != identity_of(base_cfg)


def test_the_record_is_written_before_eole_starts(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    go(config_path, run_dir, teacher.final)

    for call in fake.calls:
        assert call.launch_json is not None, call.command
        document = json.loads(call.launch_json)
        assert document["schema_version"] == 2
        assert document["continued_from"]["path"] == str(teacher.final)


def test_a_plain_launch_still_writes_a_version_1_record(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "plain.yaml", base_cfg)

    go(config, run_dir, None)

    document = record_document(run_dir)
    assert document["schema_version"] == 1
    assert "continued_from" not in document


# ---- an interrupted continuation ------------------------------------------------------------------


def test_the_next_attempt_resumes_from_the_runs_own_step_with_no_reset_optim(
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    start_and_crash(
        fake, config_path, run_dir, teacher.final, step=120
    )  # steps 50, 100

    result = go(config_path, run_dir, teacher.final)

    expected = json.loads(json.dumps(base_cfg))
    expected["training"]["train_from"] = str(run_dir / "ckpt" / "step_100")
    # Stated, not left out: the checkpoints of the run record `reset_optim: all` and Eole
    # fills an omitted key from that record, which would restart the count at step 1.
    expected["training"]["reset_optim"] = "none"
    assert rendered(run_dir, 2) == expected
    assert fake.trains[1].config == expected
    assert "reset_optim: all" not in (run_dir / "launch" / "train-2.yaml").read_text(
        encoding="utf-8"
    )
    assert (result.status, result.resumed_from, result.attempt) == ("completed", 100, 2)


def test_reset_optim_all_is_in_the_first_attempts_file_only(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final, step=120)
    go(config_path, run_dir, teacher.final)

    texts = {
        attempt: (run_dir / "launch" / f"train-{attempt}.yaml").read_text(
            encoding="utf-8"
        )
        for attempt in (1, 2)
    }
    assert "reset_optim: all" in texts[1]
    assert f"train_from: {teacher.final}" in texts[1]
    assert "reset_optim: all" not in texts[2]
    assert "reset_optim: none" in texts[2]
    assert str(teacher.final) not in texts[2]


def test_a_third_attempt_resumes_from_the_newest_step_of_the_run(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final, step=70)
    fake.queue(crash(170))
    with pytest.raises(UpstreamError):
        go(config_path, run_dir, teacher.final)

    result = go(config_path, run_dir, teacher.final)

    assert rendered(run_dir, 3)["training"]["train_from"] == str(
        run_dir / "ckpt" / "step_150"
    )
    assert rendered(run_dir, 3)["training"]["reset_optim"] == "none"
    assert (result.resumed_from, result.attempt) == (150, 3)


def test_an_attempt_that_saved_no_step_starts_from_the_source_again(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final, step=30)
    assert not any((run_dir / "ckpt").glob("step_*"))

    result = go(config_path, run_dir, teacher.final)

    second = rendered(run_dir, 2)["training"]
    assert second["train_from"] == str(teacher.final)
    assert second["reset_optim"] == "all"
    assert (result.status, result.resumed_from, result.attempt) == (
        "completed",
        None,
        2,
    )


def test_a_run_whose_only_step_directory_is_incomplete_starts_from_the_source_again(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    fake.queue(
        Script(to_step=70, cut=(50,), skip_final_save=True, error=UpstreamError(FAILED))
    )
    with pytest.raises(UpstreamError):
        go(config_path, run_dir, teacher.final)

    result = go(config_path, run_dir, teacher.final)

    second = rendered(run_dir, 2)["training"]
    assert second["train_from"] == str(teacher.final)
    assert second["reset_optim"] == "all"
    assert (result.resumed_from, result.incomplete_skipped) == (None, (50,))


def test_the_resume_does_not_change_the_record(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final)
    before = (run_dir / "launch.json").read_bytes()

    go(config_path, run_dir, teacher.final)

    assert (run_dir / "launch.json").read_bytes() == before


def test_the_source_is_verified_again_before_a_resume_starts_anything(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final)
    step_dirs.flip_one_byte(teacher.final / "model.00.safetensors")
    calls, files = len(fake.calls), files_under(run_dir)

    with pytest.raises(ConfigError, match=r"model\.00\.safetensors.*differs"):
        go(config_path, run_dir, teacher.final)

    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


def test_a_source_that_moved_is_the_same_source_when_its_files_are_the_same(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final)
    moved = tmp_path / "archive" / "run"
    moved.parent.mkdir()
    shutil.move(teacher.run_dir, moved)

    result = go(config_path, run_dir, moved / "ckpt" / "step_200")

    assert (result.status, result.resumed_from) == ("completed", 100)
    assert result.continued_from is not None
    assert result.continued_from.path == moved / "ckpt" / "step_200"
    # The record keeps where the run was started from.
    assert record_document(run_dir)["continued_from"]["path"] == str(teacher.final)


def test_another_source_for_the_same_run_is_another_run_and_names_both_sources(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final)
    other = make_finished_run(tmp_path / "other", steps=(50, 100, 150, 300))
    calls, files = len(fake.calls), files_under(run_dir)

    with pytest.raises(ConfigError) as raised:
        go(config_path, run_dir, other.final)

    message = str(raised.value)
    assert "the config is not the one this run was started with" in message
    assert str(teacher.final) in message
    assert str(other.final) in message
    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


def test_a_source_of_another_kind_is_another_source(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final)
    averaged = make_averaged_dir(tmp_path / "averaged")

    with pytest.raises(ConfigError, match="not the one this run was started with"):
        go(config_path, run_dir, averaged)


def test_another_config_for_the_same_source_is_refused_as_before(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final)
    other = write_config(
        tmp_path / "other.yaml", config_with(base_cfg, learning_rate=0.25)
    )

    with pytest.raises(ConfigError, match="not the one this run was started with"):
        go(other, run_dir, teacher.final)


def test_a_plain_launch_of_a_continuation_is_refused_naming_its_source(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final)
    calls = len(fake.calls)

    with pytest.raises(ConfigError) as raised:
        go(config_path, run_dir, None)

    assert "not the one this run was started with" in str(raised.value)
    assert str(teacher.final) in str(raised.value)
    assert len(fake.calls) == calls


def test_the_train_steps_of_a_continuation_may_be_raised(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    go(config_path, run_dir, teacher.final)
    longer = write_config(
        tmp_path / "longer.yaml", config_with(base_cfg, train_steps=500)
    )

    result = go(longer, run_dir, teacher.final)

    assert (result.status, result.steps, result.resumed_from) == ("completed", 500, 400)
    extension = rendered(run_dir, 2)["training"]
    assert extension["train_from"] == str(run_dir / "ckpt" / "step_400")
    assert extension["reset_optim"] == "none"


# ---- a run that is not a fresh one ---------------------------------------------------------------


def test_a_continuation_into_a_run_that_began_from_random_init_is_refused(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, None)  # a plain run, steps 50 and 100
    calls, files = len(fake.calls), files_under(run_dir)

    with pytest.raises(ConfigError) as raised:
        go(config_path, run_dir, teacher.final)

    message = str(raised.value)
    assert "was not started as a continuation" in message
    assert str(run_dir / "launch.json") in message
    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


def test_a_continuation_into_a_run_whose_only_step_directory_is_incomplete_is_refused_the_same(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    # A plain run that was killed while its first checkpoint was written: the step
    # directory is incomplete, and the run still began from random init.
    fake.queue(
        Script(to_step=70, cut=(50,), skip_final_save=True, error=UpstreamError(FAILED))
    )
    with pytest.raises(UpstreamError):
        go(config_path, run_dir, None)
    assert [step.complete for step in step_directories(run_dir / "ckpt")] == [False]

    with pytest.raises(ConfigError, match="was not started as a continuation"):
        go(config_path, run_dir, teacher.final)


def test_a_continuation_into_a_finished_plain_run_is_refused_too(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    go(config_path, run_dir, None)
    files = files_under(run_dir)

    with pytest.raises(ConfigError, match="was not started as a continuation"):
        go(config_path, run_dir, teacher.final)

    assert files_under(run_dir) == files


def test_step_directories_without_a_record_are_refused_as_for_any_launch(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    step_dirs.make_step_dir(run_dir / "ckpt", 50)

    with pytest.raises(ConfigError, match="does not exist"):
        go(config_path, run_dir, teacher.final)

    assert fake.calls == []


def test_a_record_without_step_directories_is_just_a_fresh_start(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    go(config_path, run_dir, None, write_only=True)  # a plain record, no steps
    assert record_document(run_dir)["schema_version"] == 1

    result = go(config_path, run_dir, teacher.final)

    assert result.status == "completed"
    assert record_document(run_dir)["schema_version"] == 2
    assert rendered(run_dir, 1)["training"]["reset_optim"] == "all"


def test_a_stepless_record_of_another_source_is_replaced(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final, step=30)
    other = make_finished_run(tmp_path / "other", steps=(50, 100, 150, 300))

    go(config_path, run_dir, other.final)

    assert record_document(run_dir)["continued_from"]["path"] == str(other.final)


# ---- write_only ------------------------------------------------------------------------------


def test_write_only_writes_the_script_the_config_and_the_record_and_runs_nothing(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    result = go(config_path, run_dir, teacher.final, write_only=True)

    assert fake.calls == []
    assert (result.status, result.attempt, result.command_sh) == (
        "written",
        1,
        run_dir / "command.sh",
    )
    assert result.continued_from == verify_source(teacher.final)
    script = (run_dir / "command.sh").read_text(encoding="utf-8")
    assert f"train -config {run_dir / 'launch' / 'train-1.yaml'}" in script
    assert rendered(run_dir)["training"]["train_from"] == str(teacher.final)
    assert rendered(run_dir)["training"]["reset_optim"] == "all"
    assert record_document(run_dir)["continued_from"]["path"] == str(teacher.final)
    assert not (run_dir / "launch" / "lock").exists()


def test_the_script_of_a_continuation_says_it_starts_from_the_source(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    go(config_path, run_dir, teacher.final, write_only=True)

    script = (run_dir / "command.sh").read_text(encoding="utf-8")
    assert f"# It continues {json.dumps(str(teacher.final))} " in script
    assert "from scratch" not in script


def test_write_only_after_a_crash_writes_the_resume_attempt(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    start_and_crash(fake, config_path, run_dir, teacher.final)

    result = go(config_path, run_dir, teacher.final, write_only=True)

    assert (result.status, result.resumed_from, result.attempt) == ("written", 100, 2)
    assert rendered(run_dir, 2)["training"]["reset_optim"] == "none"
    assert "# It resumes from step 100" in (run_dir / "command.sh").read_text(
        encoding="utf-8"
    )


def test_write_only_refuses_what_a_launch_refuses(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    step_dirs.flip_one_byte(teacher.final / "model.00.safetensors")

    with pytest.raises(ConfigError, match="differs"):
        go(config_path, run_dir, teacher.final, write_only=True)

    assert not run_dir.exists()


# ---- a finished continuation -----------------------------------------------------------------


def test_a_finished_continuation_is_found_finished_and_changes_nothing(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    first = go(config_path, run_dir, teacher.final)
    calls, files = len(fake.calls), files_under(run_dir)

    again = go(config_path, run_dir, teacher.final)

    assert again == LaunchResult(
        status="already_complete",
        steps=400,
        resumed_from=None,
        checkpoint=run_dir / "ckpt" / "step_400",
        command_sh=None,
        log=None,
        attempt=0,
        incomplete_skipped=(),
        continued_from=first.continued_from,
    )
    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


def test_a_finished_continuation_does_not_need_its_source_any_more(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    first = go(config_path, run_dir, teacher.final)
    files = files_under(run_dir)
    shutil.rmtree(teacher.run_dir)

    again = go(config_path, run_dir, teacher.final)

    assert again.status == "already_complete"
    assert files_under(run_dir) == files
    # What it says about the source is what the run recorded.
    assert again.continued_from == first.continued_from


def test_a_finished_continuation_is_judged_by_its_record_and_does_not_verify_the_source(
    config_path: Path,
    run_dir: Path,
    teacher: FabricatedRun,
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    go(config_path, run_dir, teacher.final)
    step_dirs.flip_one_byte(teacher.final / "model.00.safetensors")
    seen: list[Path] = []
    real = launch_module.verify_source
    monkeypatch.setattr(
        launch_module, "verify_source", lambda path: seen.append(path) or real(path)
    )

    again = go(config_path, run_dir, teacher.final)

    assert again.status == "already_complete"
    assert seen == []


def test_a_finished_continuation_with_another_config_is_refused_as_before(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    go(config_path, run_dir, teacher.final)
    other = write_config(tmp_path / "o.yaml", config_with(base_cfg, learning_rate=0.25))

    with pytest.raises(ConfigError, match="not the one this run was started with"):
        go(other, run_dir, teacher.final)


def test_extending_a_finished_continuation_needs_the_source(
    tmp_path: Path,
    config_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    go(config_path, run_dir, teacher.final)
    longer = write_config(
        tmp_path / "longer.yaml", config_with(base_cfg, train_steps=500)
    )
    shutil.rmtree(teacher.run_dir)
    calls, files = len(fake.calls), files_under(run_dir)

    with pytest.raises(ConfigError, match="does not exist"):
        go(longer, run_dir, teacher.final)

    assert len(fake.calls) == calls
    assert files_under(run_dir) == files


# ---- what is checked, and before what ----------------------------------------------------------


def test_a_source_that_the_manifest_does_not_vouch_for_leaves_only_the_lock(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    step_dirs.flip_one_byte(teacher.final / "config.json", marker=b"saved_at")

    with pytest.raises(ConfigError, match=r"config\.json.*differs"):
        go(config_path, run_dir, teacher.final)

    assert fake.calls == []
    assert files_under(run_dir) == {"launch/lock": b""}


def test_an_unreadable_manifest_of_the_source_is_a_data_error_before_anything_is_written(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    teacher.manifest.write_text("not json", encoding="utf-8")

    with pytest.raises(DataError, match="manifest"):
        go(config_path, run_dir, teacher.final)

    assert fake.calls == []
    assert files_under(run_dir) == {"launch/lock": b""}


def test_the_hook_that_drops_the_old_manifest_is_called_once_after_the_checks(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    hooks: list[str] = []

    def before_writing() -> None:
        # By now the source was verified and nothing of the run was written.
        assert not (run_dir / "launch" / "train-1.yaml").exists()
        hooks.append("called")

    go(config_path, run_dir, teacher.final, before_writing=before_writing)

    assert hooks == ["called"]


def test_the_hook_is_not_called_for_a_refused_source(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    step_dirs.flip_one_byte(teacher.final / "model.00.safetensors")
    hooks: list[str] = []

    with pytest.raises(ConfigError):
        go(
            config_path,
            run_dir,
            teacher.final,
            before_writing=lambda: hooks.append("x"),
        )

    assert hooks == []


def test_the_run_is_locked_before_the_source_is_looked_at(
    config_path: Path,
    run_dir: Path,
    teacher: FabricatedRun,
    fake: FakeEole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []
    real = launch_module.verify_source
    monkeypatch.setattr(
        launch_module, "verify_source", lambda path: seen.append(path) or real(path)
    )

    with (
        holding_the_run(run_dir / "launch" / "lock"),
        pytest.raises(ConfigError, match="lock"),
    ):
        go(config_path, run_dir, teacher.final)

    assert seen == []
    assert fake.calls == []


def test_write_only_is_refused_while_another_launch_holds_the_run(
    config_path: Path, run_dir: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    with (
        holding_the_run(run_dir / "launch" / "lock"),
        pytest.raises(ConfigError, match="lock"),
    ):
        go(config_path, run_dir, teacher.final, write_only=True)

    assert not (run_dir / "launch.json").exists()


@pytest.mark.parametrize("key", ["train_from", "reset_optim"])
def test_a_base_config_that_holds_either_key_is_refused_even_for_a_continuation(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
    key: str,
) -> None:
    cfg = config_with(
        base_cfg, **{key: "all" if key == "reset_optim" else str(teacher.final)}
    )
    config = write_config(tmp_path / "smuggled.yaml", cfg)

    with pytest.raises(ConfigError, match=f"training.{key}: unknown key in training"):
        go(config, run_dir, teacher.final)

    assert not run_dir.exists()
    assert fake.calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"heads": 8}, r"model\.heads is 4 in the checkpoint but 8 in the config"),
        (
            {"hidden_size": 128, "embeddings": {"word_vec_size": 128}},
            r"model\.hidden_size is 64",
        ),
    ],
)
def test_a_config_for_another_architecture_than_the_sources_is_refused(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
    change: dict[str, Any],
    message: str,
) -> None:
    cfg = json.loads(json.dumps(base_cfg))
    for key, value in change.items():
        if isinstance(value, dict):
            cfg["model"][key].update(value)
        else:
            cfg["model"][key] = value
    config = write_config(tmp_path / "other-model.yaml", cfg)

    with pytest.raises(ConfigError, match=message):
        go(config, run_dir, teacher.final)

    assert fake.calls == []
    assert files_under(run_dir) == {"launch/lock": b""}


def test_a_config_that_leaves_an_optimizer_option_to_the_source_is_refused(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    cfg = json.loads(json.dumps(base_cfg))
    del cfg["training"]["adam_beta2"]
    config = write_config(tmp_path / "open.yaml", cfg)

    with pytest.raises(ConfigError, match=r"training\.adam_beta2"):
        go(config, run_dir, teacher.final)

    assert fake.calls == []


def test_a_teacher_without_the_reserved_symbols_is_refused_before_eole_starts(
    tmp_path: Path, config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    old = make_finished_run(
        tmp_path / "old",
        vocab_tokens=[t for t in SOURCE_VOCABULARY if t not in ("<doc>", "<sep>")],
    )

    with pytest.raises(ConfigError, match="predates the vocabulary invariant"):
        go(config_path, run_dir, old.final)

    assert fake.calls == []
    assert files_under(run_dir) == {"launch/lock": b""}


# ---- a path that must never reach Eole -----------------------------------------------------------


@pytest.mark.parametrize("char", ["$", "[", "*", "?"])
def test_a_source_path_with_a_variable_or_glob_character_writes_nothing_at_all(
    tmp_path: Path, config_path: Path, run_dir: Path, fake: FakeEole, char: str
) -> None:
    source = make_finished_run(tmp_path / f"t{char}x")

    with pytest.raises(ConfigError, match="continuation source"):
        go(config_path, run_dir, source.final)

    assert not run_dir.exists()
    assert fake.calls == []


def test_a_source_inside_the_new_run_writes_nothing(
    tmp_path: Path, fake: FakeEole
) -> None:
    run_dir = tmp_path / "inside"
    inside = make_finished_run(run_dir / "old")
    config = write_config(tmp_path / "base.yaml", continuation_config(run_dir))

    with pytest.raises(ConfigError, match="inside"):
        go(config, run_dir, inside.final)

    assert not (run_dir / "launch").exists()
    assert fake.calls == []


def test_the_new_runs_own_checkpoint_directory_is_not_a_source(
    config_path: Path, run_dir: Path, fake: FakeEole
) -> None:
    step_dirs.make_step_dir(run_dir / "ckpt", 200)

    with pytest.raises(ConfigError, match="inside"):
        go(config_path, run_dir, run_dir / "ckpt")
    with pytest.raises(ConfigError, match="inside"):
        go(config_path, run_dir, run_dir / "ckpt" / "step_200")

    assert fake.calls == []
    assert not (run_dir / "launch").exists()


def test_a_run_directory_inside_the_source_is_refused(
    tmp_path: Path, teacher: FabricatedRun, fake: FakeEole
) -> None:
    new_run = teacher.final / "continued"
    config = write_config(tmp_path / "base.yaml", continuation_config(new_run))
    before = tree_digest(teacher.run_dir)

    with pytest.raises(ConfigError, match="inside the source"):
        go(config, new_run, teacher.final)

    assert tree_digest(teacher.run_dir) == before


def test_a_vocabulary_file_inside_the_source_is_refused(
    tmp_path: Path,
    run_dir: Path,
    base_cfg: dict[str, Any],
    teacher: FabricatedRun,
    fake: FakeEole,
) -> None:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["src_vocab"] = str(teacher.final / "vocab.src")
    config = write_config(tmp_path / "base.yaml", cfg)
    before = tree_digest(teacher.run_dir)

    with pytest.raises(ConfigError, match="vocabulary"):
        go(config, run_dir, teacher.final)

    assert tree_digest(teacher.run_dir) == before
    assert fake.calls == []


# ---- a plain run is not touched ---------------------------------------------------------


def test_a_plain_run_resumes_with_no_reset_optim_at_all(
    tmp_path: Path, run_dir: Path, base_cfg: dict[str, Any], fake: FakeEole
) -> None:
    config = write_config(tmp_path / "plain.yaml", base_cfg)
    start_and_crash(fake, config, run_dir, None)

    go(config, run_dir, None)

    assert "reset_optim" not in rendered(run_dir, 2)["training"]
    assert "reset_optim" not in rendered(run_dir, 1)["training"]
