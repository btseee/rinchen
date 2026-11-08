"""Tests for `rinchen.train.ablation_records`: when a recorded `result.json` makes a
variant complete, so that a rerun reuses it, and when it is never trusted.

The variants are run through `run_ablation` with the fakes of
`tests/support/fake_ablation.py` (no Eole); each test changes one input or one byte
of a record and says which variants train again.
"""

import dataclasses
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rinchen.train import ablation_prepare
from rinchen.train.ablation_plan import AblationPlan, plan_ablation
from tests.support.corpus_rows import EN_MN
from tests.support.fake_ablation import (
    ID_SRC,
    ID_TGT,
    AblationRun,
    flip_a_byte,
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
def other_spm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_toy_spm(tmp_path_factory.mktemp("ablation_other_spm"), vocab_size=600)


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


Run = AblationRun


# --- run_ablation: what makes a variant complete -------------------------------------


def test_a_changed_bt_row_reruns_only_the_variants_that_use_bt_rows(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    axes = {"temperature": [0.5], "bt_ratio": [0.0, 3.0], "tag_placement": ["src"]}
    first = toy_plan(tmp_path, toy_spm, corpus=corpus, axes=axes)
    Run(first)
    changed = write_toy_corpus(tmp_path / "changed", changed_bt_row=True)
    second = plan_ablation(make_matrix(axes=axes), [changed], toy_spm, first.out_dir)

    run = Run(second)

    assert run.train.variant_ids == ["t0.5_bt3_src_s1234"]


def test_another_sentencepiece_model_reruns_every_variant(
    default_plan: AblationPlan, corpus: Path, other_spm: Path, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)
    changed = plan_ablation(make_matrix(), [corpus], other_spm, plan.out_dir)

    run = Run(changed)

    assert run.train.variant_ids == [ID_SRC, ID_TGT]


def test_a_changed_override_reruns_every_variant(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    first = toy_plan(tmp_path, toy_spm, corpus=corpus)
    Run(first)
    second = plan_ablation(
        make_matrix(overrides={"training": {"dropout": [0.1]}}),
        [corpus],
        toy_spm,
        first.out_dir,
    )

    run = Run(second)

    assert run.train.variant_ids == [ID_SRC, ID_TGT]
    assert "0.1" in run.train.calls[0].config_text


def test_another_thread_count_reruns_every_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan, threads=2)

    run = Run(plan, threads=3)

    assert run.train.variant_ids == [ID_SRC, ID_TGT]
    assert {call.threads for call in run.train.calls} == {3}
    assert {call.threads for call in run.evaluate.calls} == {3}
    assert run.result.threads == 3
    document = json.loads((plan.out_dir / "results.json").read_text(encoding="utf-8"))
    assert document["threads"] == 3
    assert Run(plan, threads=3).train.calls == []


def test_another_eole_version_reruns_every_variant(
    default_plan: AblationPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan)
    monkeypatch.setattr(
        ablation_prepare, "tool_versions", lambda packages: {"eole": "9.9.9"}
    )

    run = Run(plan)

    assert run.train.variant_ids == [ID_SRC, ID_TGT]


def test_a_different_timeout_does_not_rerun_a_complete_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    Run(plan, timeout=30.0)

    run = Run(plan, timeout=None)

    assert run.train.calls == []


def test_a_modified_checkpoint_reruns_exactly_its_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    flip_a_byte(first.result.variants[0].checkpoint_path)

    run = Run(plan)

    assert run.train.variant_ids == [ID_SRC]
    assert Run(plan).train.calls == []


@pytest.mark.parametrize("how", ["file", "directory", "empty"])
def test_a_missing_checkpoint_reruns_exactly_its_variant(
    default_plan: AblationPlan, tmp_path: Path, how: str
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    checkpoint = first.result.variants[1].checkpoint_path
    if how == "file":
        checkpoint.unlink()
    elif how == "directory":
        shutil.rmtree(checkpoint.parent)
    else:
        checkpoint.write_bytes(b"")

    run = Run(plan)

    assert run.train.variant_ids == [ID_TGT]


def test_a_missing_result_file_reruns_exactly_its_variant(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    first.result_path(ID_SRC).unlink()

    run = Run(plan)

    assert run.train.variant_ids == [ID_SRC]


def _tamper(path: Path, change: Callable[[dict[str, Any]], Any]) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(change(record)), encoding="utf-8")


def _set(**fields: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda record: {**record, **fields}


def _without(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda record: {k: v for k, v in record.items() if k != key}


def _dev_with(**fields: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda record: {**record, "dev": {**record["dev"], **fields}}


def _direction_with(**fields: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def change(record: dict[str, Any]) -> dict[str, Any]:
        directions = {k: dict(v) for k, v in record["dev"]["directions"].items()}
        directions[EN_MN].update(fields)
        return {**record, "dev": {**record["dev"], "directions": directions}}

    return change


def _drop_a_direction(record: dict[str, Any]) -> dict[str, Any]:
    directions = dict(record["dev"]["directions"])
    del directions[EN_MN]
    return {**record, "dev": {**record["dev"], "directions": directions}}


def _add_a_direction(record: dict[str, Any]) -> dict[str, Any]:
    directions = dict(record["dev"]["directions"])
    directions["eng_Latn-rus_Cyrl"] = directions[EN_MN]
    return {**record, "dev": {**record["dev"], "directions": directions}}


UNTRUSTED_CHANGES: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
    ("another fingerprint", _set(fingerprint="0" * 64)),
    ("fingerprint of the wrong type", _set(fingerprint=5)),
    ("no fingerprint", _without("fingerprint")),
    ("status failed", _set(status="failed")),
    ("status true", _set(status=True)),
    ("no status", _without("status")),
    ("another checkpoint hash", _set(checkpoint_sha256="f" * 64)),
    ("checkpoint hash of the wrong type", _set(checkpoint_sha256=7)),
    ("no checkpoint hash", _without("checkpoint_sha256")),
    ("train seconds as text", _set(train_seconds="1.5")),
    ("negative train seconds", _set(train_seconds=-1.0)),
    ("train seconds a boolean", _set(train_seconds=True)),
    ("no train seconds", _without("train_seconds")),
    ("no eval seconds", _without("eval_seconds")),
    ("negative eval seconds", _set(eval_seconds=-0.5)),
    ("no dev result", _without("dev")),
    ("dev result a list", _set(dev=[])),
    ("dev result null", _set(dev=None)),
    ("leaks as text", _dev_with(tag_leaks="0")),
    ("leaks a boolean", _dev_with(tag_leaks=True)),
    ("leaks a float", _dev_with(tag_leaks=0.5)),
    ("chrf mean null", _dev_with(chrf_mean=None)),
    ("chrf min missing", _dev_with(chrf_min=None)),
    ("signature of the wrong type", _dev_with(chrf_signature=3)),
    ("chrf mean not a number", _dev_with(chrf_mean=float("nan"))),
    ("chrf in a direction infinite", _direction_with(chrf=float("inf"))),
    ("train seconds infinite", _set(train_seconds=float("inf"))),
    ("eval seconds not a number", _set(eval_seconds=float("nan"))),
    ("chrf as text", _direction_with(chrf="80.0")),
    ("bleu a boolean", _direction_with(bleu=True)),
    ("lines as a float", _direction_with(lines=100.5)),
    ("another number of lines", _direction_with(lines=99)),
    ("a direction missing", _drop_a_direction),
    ("a direction too many", _add_a_direction),
    ("a list", lambda record: [record]),
    ("null", lambda record: None),
    ("text", lambda record: "ok"),
    ("a number", lambda record: 5),
    ("an empty object", lambda record: {}),
    ("another schema", lambda record: {"status": "ok", "fingerprint": "abc"}),
]


@pytest.mark.parametrize(
    "change",
    [item[1] for item in UNTRUSTED_CHANGES],
    ids=[i[0] for i in UNTRUSTED_CHANGES],
)
def test_a_result_file_that_does_not_vouch_for_the_variant_is_never_trusted(
    default_plan: AblationPlan, tmp_path: Path, change: Callable[[dict[str, Any]], Any]
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    _tamper(first.result_path(ID_TGT), change)

    run = Run(plan)

    assert run.train.variant_ids == [ID_TGT]
    assert (
        json.loads(run.result_path(ID_TGT).read_text(encoding="utf-8"))["status"]
        == "ok"
    )
    assert Run(plan).train.calls == []


def test_a_result_file_with_unknown_extra_keys_is_still_trusted(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    _tamper(first.result_path(ID_TGT), _set(note="added by hand"))

    assert Run(plan).train.calls == []


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"{",
        b"not json",
        b"\xff\xfe\x00\x01",
        b"[" * 100_000,
        b'{"status": "ok", "fingerprint": NaN}',
        b"\x00" * 64,
    ],
    ids=["empty", "open brace", "text", "not utf-8", "deeply nested", "nan", "nul"],
)
def test_a_result_file_that_is_not_json_is_never_trusted_and_never_a_crash(
    default_plan: AblationPlan, tmp_path: Path, content: bytes
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    first.result_path(ID_SRC).write_bytes(content)

    run = Run(plan)

    assert run.train.variant_ids == [ID_SRC]


def test_a_truncated_result_file_is_never_trusted(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    text = first.result_path(ID_SRC).read_bytes()
    first.result_path(ID_SRC).write_bytes(text[: len(text) // 2])

    assert Run(plan).train.variant_ids == [ID_SRC]


def test_a_directory_in_place_of_the_result_file_is_an_os_error_not_a_reuse(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)
    first = Run(plan)
    path = first.result_path(ID_SRC)
    path.unlink()
    path.mkdir()

    with pytest.raises(OSError):
        Run(plan)
