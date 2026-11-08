"""Tests for `rinchen.train.ablation`: the run that trains and evaluates the variants
one after the other, reuses the complete ones and writes the result tables.

Nothing starts Eole: `train` and `evaluate` are the fakes of
`tests/support/fake_ablation.py`, which record what they are handed, write a
checkpoint file and return scripted dev scores. The plan's numbers are worked out in
`tests/unit/test_train_ablation_plan.py`; when a variant counts as complete is tested
in `tests/unit/test_train_ablation_records.py`.
"""

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.train.ablation import run_ablation
from rinchen.train.ablation_plan import AblationPlan, plan_ablation
from rinchen.train.ablation_prepare import prepare_variant
from rinchen.train.ablation_records import VariantProgress
from rinchen.train.ablation_report import result_files
from rinchen.train.eole_cfg import validate_train_config
from rinchen.train.eole_run import CHECKPOINT_FILE, run_training
from rinchen.train.evaluate import DevResult, evaluate_variant
from rinchen.train.variants import expand_variants
from tests.support.corpus_rows import EN_MN
from tests.support.fake_ablation import (
    DIRECTIONS,
    ID_SRC,
    ID_TGT,
    AblationRun,
    FakeClock,
    FakeEvaluate,
    FakeTrain,
    ProgressLog,
    dev_result,
    make_matrix,
    toy_plan,
    write_toy_corpus,
)
from tests.support.tiny_parallel import train_toy_spm

# (name, rows, weight, probability, upsampling) of the default variant, in the order
# of the plan: directions sorted, `auth` before `bt`.


@pytest.fixture(scope="module")
def toy_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("ablation_spm"))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_toy_corpus(tmp_path_factory.mktemp("ablation_corpus"))


@pytest.fixture(scope="module")
def default_plan(
    corpus: Path, toy_spm: Path, tmp_path_factory: pytest.TempPathFactory
) -> AblationPlan:
    """The plan of the default matrix (src and tgt at temperature 0.5, bt 0.5). Its
    output directory is never created: tests take it through `_plan_in`."""
    out_dir = tmp_path_factory.mktemp("ablation_default") / "out"
    return plan_ablation(make_matrix(), [corpus], toy_spm, out_dir)


def _plan_in(plan: AblationPlan, tmp_path: Path) -> AblationPlan:
    return dataclasses.replace(plan, out_dir=(tmp_path / "out").resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


Run = AblationRun


@pytest.fixture(scope="module")
def completed(
    default_plan: AblationPlan, tmp_path_factory: pytest.TempPathFactory
) -> Run:
    """A finished run of the default plan, for tests that only read what it wrote."""
    plan = _plan_in(default_plan, tmp_path_factory.mktemp("ablation_completed"))
    return Run(plan)


# --- run_ablation: sequencing --------------------------------------------------------


def test_the_seams_default_to_the_real_training_and_evaluation() -> None:
    parameters = inspect.signature(run_ablation).parameters

    assert parameters["train"].default is run_training
    assert parameters["evaluate"].default is evaluate_variant
    assert parameters["progress"].default is None
    for name in ("threads", "timeout", "force"):
        assert parameters[name].default is inspect.Parameter.empty


def test_the_variants_run_in_the_order_of_the_expansion(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes={
            "temperature": [1.0, 0.5],
            "bt_ratio": [0.0],
            "tag_placement": ["tgt", "src"],
        },
        seeds=[2, 1],
    )

    run = Run(plan)

    expected = [v.id for v in expand_variants(plan.matrix)]
    assert expected[:3] == ["t1_bt0_tgt_s2", "t1_bt0_tgt_s1", "t1_bt0_src_s2"]
    assert run.train.variant_ids == expected
    assert run.evaluate.variant_ids == expected
    assert [r.planned.variant.id for r in run.result.variants] == expected


def test_training_is_given_the_config_the_run_directory_the_validator_and_the_limits(
    completed: Run,
) -> None:
    calls = {call.variant_id: call for call in completed.train.calls}

    for variant_id in (ID_SRC, ID_TGT):
        call = calls[variant_id]
        directory = completed.variant_dir(variant_id)
        assert call.cfg_path == directory / "config.yaml"
        assert call.run_dir == directory / "run"
        assert call.validate is validate_train_config
        assert (call.threads, call.timeout) == (2, 30.0)
        assert call.config_text == (directory / "config.yaml").read_text(
            encoding="utf-8"
        )


def test_the_evaluation_is_given_the_checkpoint_the_dev_sets_and_the_placement(
    completed: Run,
) -> None:
    calls = {call.variant_id: call for call in completed.evaluate.calls}

    for variant_id, placement in ((ID_SRC, "src"), (ID_TGT, "tgt")):
        call = calls[variant_id]
        directory = completed.variant_dir(variant_id)
        assert call.ckpt_dir == directory / "run" / "ckpt"
        assert call.dev is completed.plan.dev.sets
        assert call.spm_model == completed.plan.spm_model
        assert call.placement == placement
        assert call.out_dir == directory / "eval"
        assert (call.threads, call.timeout) == (2, 30.0)


def test_the_result_file_is_written_after_the_evaluation_and_not_before(
    completed: Run,
) -> None:
    assert [call.result_existed for call in completed.evaluate.calls] == [False, False]
    for variant_id in (ID_SRC, ID_TGT):
        assert completed.result_path(variant_id).is_file()


def test_the_result_of_a_variant_is_what_the_fakes_returned_plus_the_checkpoint_hash(
    completed: Run,
) -> None:
    src, tgt = completed.result.variants

    assert (src.planned.variant.id, tgt.planned.variant.id) == (ID_SRC, ID_TGT)
    assert src.dev == dev_result(80.0)
    assert not src.reused and not tgt.reused
    for result in (src, tgt):
        directory = completed.variant_dir(result.planned.variant.id)
        file = directory / "run" / "ckpt" / CHECKPOINT_FILE
        assert result.checkpoint_path == file
        assert result.checkpoint_sha256 == _sha256(file)
        assert (
            result.fingerprint
            == prepare_variant(completed.plan, result.planned, threads=2).fingerprint
        )


def test_the_result_file_records_the_variant_the_scores_and_the_checkpoint_hash(
    completed: Run,
) -> None:
    record = json.loads(completed.result_path(ID_TGT).read_text(encoding="utf-8"))
    result = completed.result.variants[1]

    assert record["status"] == "ok"
    assert record["id"] == ID_TGT
    assert record["tag_placement"] == "tgt"
    assert record["fingerprint"] == result.fingerprint
    assert record["checkpoint_sha256"] == result.checkpoint_sha256
    assert record["checkpoint_file"] == result.checkpoint_path.as_posix()
    assert record["dev"] == result.dev.to_dict()
    weights = {c["name"]: c["weight"] for c in record["corpora"]}
    assert weights[f"{EN_MN}.auth"] == 22
    assert sum(weights.values()) == 100
    assert completed.result_path(ID_TGT).read_text(encoding="utf-8").endswith("}\n")


def test_the_durations_are_those_of_the_clock_around_each_call(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    clock = FakeClock()
    run = Run(
        _plan_in(default_plan, tmp_path),
        train=FakeTrain(clock=clock, seconds={ID_SRC: 12.5, ID_TGT: 20.25}),
        evaluate=FakeEvaluate(clock=clock, seconds={ID_SRC: 3.0, ID_TGT: 4.75}),
        clock=clock,
    )

    assert [(r.train_seconds, r.eval_seconds) for r in run.result.variants] == [
        (12.5, 3.0),
        (20.25, 4.75),
    ]
    record = json.loads(run.result_path(ID_TGT).read_text(encoding="utf-8"))
    assert (record["train_seconds"], record["eval_seconds"]) == (20.25, 4.75)


def test_progress_is_reported_once_per_variant_after_its_result_is_written(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    clock = FakeClock()
    seen: list[tuple[VariantProgress, bool]] = []

    def watch(event: VariantProgress) -> None:
        written = (
            plan.out_dir / "variants" / event.variant_id / "result.json"
        ).exists()
        seen.append((event, written))

    Run(
        plan,
        train=FakeTrain(clock=clock, seconds={ID_SRC: 12.5, ID_TGT: 20.25}),
        evaluate=FakeEvaluate(clock=clock, seconds={ID_SRC: 3.0, ID_TGT: 4.75}),
        clock=clock,
        progress=watch,
    )

    assert seen == [
        (VariantProgress(ID_SRC, 1, 2, "trained", 12.5, 3.0), True),
        (VariantProgress(ID_TGT, 2, 2, "trained", 20.25, 4.75), True),
    ]


def test_a_progress_event_is_frozen_and_names_its_fields() -> None:
    event = VariantProgress("v", 1, 2, "reused", 0.0, 0.0)

    assert dataclasses.asdict(event) == {
        "variant_id": "v",
        "index": 1,
        "total": 2,
        "outcome": "reused",
        "train_seconds": 0.0,
        "eval_seconds": 0.0,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.index = 3  # type: ignore[misc]


def test_the_threads_must_be_at_least_one_and_nothing_runs_without_them(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    train, evaluate = FakeTrain(), FakeEvaluate()
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(ValueError, match="threads"):
        run_ablation(
            plan, threads=0, timeout=None, force=False, train=train, evaluate=evaluate
        )

    assert not train.calls and not evaluate.calls
    assert not plan.out_dir.exists()


# --- run_ablation: resuming ----------------------------------------------------------


def test_a_second_run_reuses_every_complete_variant_and_calls_neither_seam(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    log = ProgressLog()

    second = Run(plan, progress=log)

    assert not second.train.calls and not second.evaluate.calls
    assert log.outcomes == [(ID_SRC, "reused"), (ID_TGT, "reused")]
    assert [(e.train_seconds, e.eval_seconds) for e in log.events] == [(0.0, 0.0)] * 2
    assert [e.index for e in log.events] == [1, 2]
    assert [r.reused for r in second.result.variants] == [True, True]
    for old, new in zip(first.result.variants, second.result.variants, strict=True):
        assert dataclasses.replace(new, outcome="trained") == old


def test_force_reruns_every_variant(default_plan: AblationPlan, tmp_path: Path) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)
    log = ProgressLog()

    again = Run(plan, force=True, progress=log)

    assert again.train.variant_ids == [ID_SRC, ID_TGT]
    assert log.outcomes == [(ID_SRC, "trained"), (ID_TGT, "trained")]
    assert [r.reused for r in again.result.variants] == [False, False]


def test_a_failing_training_stops_the_run_and_leaves_the_earlier_variants_complete(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    failure = UpstreamError("eole train exited with code 1")
    log = ProgressLog()

    with pytest.raises(UpstreamError, match="eole train exited with code 1") as info:
        Run(plan, train=FakeTrain(fail={ID_TGT: failure}), progress=log)

    assert info.value is failure
    assert log.outcomes == [(ID_SRC, "trained")]
    variants = plan.out_dir / "variants"
    assert (variants / ID_SRC / "result.json").is_file()
    assert not (variants / ID_TGT / "result.json").exists()
    assert not (plan.out_dir / "results.tsv").exists()

    resumed = Run(plan, progress=(again := ProgressLog()))

    assert resumed.train.variant_ids == [ID_TGT]
    assert resumed.evaluate.variant_ids == [ID_TGT]
    assert again.outcomes == [(ID_SRC, "reused"), (ID_TGT, "trained")]


def _train_record(plan: AblationPlan, variant_id: str) -> dict[str, Any]:
    path = plan.out_dir / "variants" / variant_id / "train.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_a_failing_evaluation_leaves_the_checkpoint_and_a_record_of_the_training(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    failure = DataError("no dev lines")
    clock = FakeClock()
    train = FakeTrain(clock=clock, seconds={ID_SRC: 12.5})

    with pytest.raises(DataError) as info:
        Run(
            plan,
            train=train,
            evaluate=FakeEvaluate(fail={ID_SRC: failure}, clock=clock),
            clock=clock,
        )

    assert info.value is failure
    variant = plan.out_dir / "variants" / ID_SRC
    assert not (variant / "result.json").exists()
    assert not (plan.out_dir / "variants" / ID_TGT).exists()
    checkpoint = variant / "run" / "ckpt" / CHECKPOINT_FILE
    assert checkpoint.is_file()
    assert _train_record(plan, ID_SRC) == {
        "status": "trained",
        "fingerprint": prepare_variant(plan, plan.variants[0], threads=2).fingerprint,
        "checkpoint_sha256": _sha256(checkpoint),
        "train_seconds": 12.5,
    }


def test_the_train_record_is_written_before_the_evaluation_starts(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    run = Run(plan)

    assert [call.train_record_existed for call in run.evaluate.calls] == [True, True]


def test_the_rerun_after_a_failing_evaluation_evaluates_the_trained_checkpoint_only(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    clock = FakeClock()
    with pytest.raises(DataError):
        Run(
            plan,
            train=FakeTrain(clock=clock, seconds={ID_SRC: 12.5}),
            evaluate=FakeEvaluate(
                fail={ID_SRC: DataError("no dev lines")}, clock=clock
            ),
            clock=clock,
        )
    checkpoint = plan.out_dir / "variants" / ID_SRC / "run" / "ckpt" / CHECKPOINT_FILE
    trained_bytes = checkpoint.read_bytes()
    log = ProgressLog()

    resumed = Run(
        plan,
        train=FakeTrain(clock=clock, seconds={ID_TGT: 20.0}),
        evaluate=FakeEvaluate(clock=clock, seconds={ID_SRC: 3.0, ID_TGT: 4.0}),
        clock=clock,
        progress=log,
    )

    assert resumed.train.variant_ids == [ID_TGT]
    assert resumed.evaluate.variant_ids == [ID_SRC, ID_TGT]
    assert log.outcomes == [(ID_SRC, "evaluated"), (ID_TGT, "trained")]
    # This call trained nothing for the first variant; the result keeps the time the
    # training took when it ran.
    assert [(e.train_seconds, e.eval_seconds) for e in log.events] == [
        (0.0, 3.0),
        (20.0, 4.0),
    ]
    first = resumed.result.variants[0]
    assert (first.train_seconds, first.eval_seconds) == (12.5, 3.0)
    assert first.outcome == "evaluated" and not first.reused
    assert first.checkpoint_sha256 == _sha256(checkpoint)
    assert checkpoint.read_bytes() == trained_bytes
    assert (plan.out_dir / "variants" / ID_SRC / "result.json").is_file()


def test_an_interrupt_during_the_evaluation_keeps_the_trained_checkpoint_as_well(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        Run(plan, evaluate=FakeEvaluate(fail={ID_SRC: KeyboardInterrupt()}))
    assert (plan.out_dir / "variants" / ID_SRC / "train.json").is_file()

    resumed = Run(plan, progress=(log := ProgressLog()))

    assert resumed.train.variant_ids == [ID_TGT]
    assert log.outcomes == [(ID_SRC, "evaluated"), (ID_TGT, "trained")]


def test_an_evaluation_that_scores_other_dev_sets_also_leaves_the_record(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    with pytest.raises(UpstreamError, match=ID_SRC):
        Run(plan, evaluate=FakeEvaluate({ID_SRC: dev_result({EN_MN: 80.0})}))

    resumed = Run(plan)

    assert resumed.train.variant_ids == [ID_TGT]
    assert resumed.evaluate.variant_ids == [ID_SRC, ID_TGT]


def test_a_training_that_fails_leaves_no_record_and_removes_an_older_one(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    with pytest.raises(DataError):
        Run(plan, evaluate=FakeEvaluate(fail={ID_SRC: DataError("no dev lines")}))
    record = plan.out_dir / "variants" / ID_SRC / "train.json"
    assert record.is_file()

    with pytest.raises(UpstreamError):
        Run(plan, force=True, train=FakeTrain(fail={ID_SRC: UpstreamError("x")}))

    assert not record.exists()
    assert not (record.parent / "result.json").exists()


def test_a_training_that_leaves_no_checkpoint_leaves_no_record(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(UpstreamError, match=CHECKPOINT_FILE):
        Run(plan, train=FakeTrain(write_checkpoint=False))

    assert not (plan.out_dir / "variants" / ID_SRC / "train.json").exists()


def test_force_trains_again_whatever_the_record_says_and_replaces_it(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    with pytest.raises(DataError):
        Run(plan, evaluate=FakeEvaluate(fail={ID_SRC: DataError("no dev lines")}))
    old = _train_record(plan, ID_SRC)

    forced = Run(
        plan,
        force=True,
        train=FakeTrain(salt=" again"),
        progress=(log := ProgressLog()),
    )

    assert forced.train.variant_ids == [ID_SRC, ID_TGT]
    assert log.outcomes == [(ID_SRC, "trained"), (ID_TGT, "trained")]
    new = _train_record(plan, ID_SRC)
    assert new["checkpoint_sha256"] != old["checkpoint_sha256"]
    assert new["fingerprint"] == old["fingerprint"]


def test_a_complete_variant_keeps_its_train_record_and_is_still_reused(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    record = _train_record(plan, ID_SRC)

    second = Run(plan, progress=(log := ProgressLog()))

    assert not second.train.calls and not second.evaluate.calls
    assert log.outcomes == [(ID_SRC, "reused"), (ID_TGT, "reused")]
    assert _train_record(plan, ID_SRC) == record
    assert record["checkpoint_sha256"] == first.result.variants[0].checkpoint_sha256


@pytest.mark.parametrize(
    "error",
    [
        UpstreamError("eole is not installed"),
        ConfigError("bad config"),
        DataError("bad data"),
        OSError("disk full"),
        RuntimeError("bug"),
    ],
)
def test_an_error_of_training_or_evaluation_reaches_the_caller_as_it_is(
    default_plan: AblationPlan, tmp_path: Path, error: Exception
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(type(error)) as from_train:
        Run(plan, train=FakeTrain(fail={ID_SRC: error}))
    with pytest.raises(type(error)) as from_evaluate:
        Run(plan, evaluate=FakeEvaluate(fail={ID_SRC: error}))

    assert from_train.value is error
    assert from_evaluate.value is error


@pytest.mark.parametrize(
    "scored",
    [
        dev_result({EN_MN: 80.0}),
        dev_result(dict.fromkeys(DIRECTIONS, 80.0), lines=99),
        dev_result({**dict.fromkeys(DIRECTIONS, 80.0), "eng_Latn-rus_Cyrl": 80.0}),
    ],
    ids=["a direction missing", "another number of lines", "a direction too many"],
)
def test_an_evaluation_that_does_not_score_the_dev_sets_of_the_plan_is_an_upstream_error(
    default_plan: AblationPlan, tmp_path: Path, scored: DevResult
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(UpstreamError, match=ID_SRC):
        Run(plan, evaluate=FakeEvaluate({ID_SRC: scored}))

    assert not (plan.out_dir / "variants" / ID_SRC / "result.json").exists()


def test_an_interrupt_stops_the_run_and_leaves_no_result_for_the_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    evaluate = FakeEvaluate()

    with pytest.raises(KeyboardInterrupt):
        Run(
            plan, train=FakeTrain(fail={ID_SRC: KeyboardInterrupt()}), evaluate=evaluate
        )

    assert not evaluate.calls
    assert not (plan.out_dir / "variants" / ID_SRC / "result.json").exists()


def test_a_forced_rerun_that_fails_leaves_no_result_of_the_earlier_run(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)

    with pytest.raises(UpstreamError):
        Run(plan, force=True, train=FakeTrain(fail={ID_SRC: UpstreamError("x")}))

    variants = plan.out_dir / "variants"
    assert not (variants / ID_SRC / "result.json").exists()
    assert (variants / ID_TGT / "result.json").is_file()


def test_a_training_that_leaves_no_checkpoint_file_is_an_upstream_error(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    evaluate = FakeEvaluate()

    with pytest.raises(UpstreamError, match=CHECKPOINT_FILE):
        Run(plan, train=FakeTrain(write_checkpoint=False), evaluate=evaluate)

    assert not evaluate.calls
    assert not (plan.out_dir / "variants" / ID_SRC / "result.json").exists()


def test_an_output_directory_that_cannot_be_made_is_an_os_error(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    plan = dataclasses.replace(default_plan, out_dir=blocker / "out")
    train = FakeTrain()

    with pytest.raises(OSError):
        Run(plan, train=train)

    assert not train.calls


# --- run_ablation: the tables --------------------------------------------------------


def test_the_run_writes_the_three_result_files_under_the_out_dir(
    completed: Run,
) -> None:
    for name in ("results.tsv", "results.json", "results.md"):
        assert (completed.plan.out_dir / name).is_file()
    assert not list(completed.plan.out_dir.rglob("*.part"))


def test_the_result_files_are_written_again_when_every_variant_is_reused(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)
    names = ("results.tsv", "results.json", "results.md")
    before = {name: (plan.out_dir / name).read_bytes() for name in names}
    for name in names:
        (plan.out_dir / name).unlink()

    again = Run(plan)

    assert again.train.calls == []
    assert {name: (plan.out_dir / name).read_bytes() for name in names} == before


def test_a_failure_to_write_the_tables_is_an_os_error_and_keeps_the_variants(
    default_plan: AblationPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    def fail(path: Path, content: bytes) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr("rinchen.train.ablation_report.replace_file", fail)
        with pytest.raises(OSError, match="disk full"):
            Run(plan)

    assert Run(plan).train.calls == []


def test_the_checkpoint_hashes_are_exposed_for_the_manifest(completed: Run) -> None:
    hashes = {
        r.planned.variant.id: r.checkpoint_sha256 for r in completed.result.variants
    }

    assert hashes[ID_SRC] == _sha256(
        completed.variant_dir(ID_SRC) / "run" / "ckpt" / CHECKPOINT_FILE
    )
    assert hashes[ID_SRC] != hashes[ID_TGT]


def test_the_result_names_the_files_it_wrote(completed: Run) -> None:
    files = result_files(completed.plan.out_dir)

    assert files.tsv == completed.plan.out_dir / "results.tsv"
    assert files.json == completed.plan.out_dir / "results.json"
    assert files.md == completed.plan.out_dir / "results.md"
