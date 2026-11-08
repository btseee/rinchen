"""Tests for `rinchen.train.ablation_prepare`: the files of one variant (corpora,
validation corpus, rendered Eole config) and the fingerprint of the inputs they were
made from.

Nothing starts Eole. The toy corpus is written as Parquet in a temporary directory;
the plan's numbers are worked out in `tests/unit/test_train_ablation_plan.py`.
"""

import dataclasses
import hashlib
import importlib.metadata
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError
from rinchen.train import ablation_prepare
from rinchen.train.ablation_plan import AblationPlan, plan_ablation
from rinchen.train.ablation_prepare import PreparedVariant, prepare_variant
from rinchen.train.eole_cfg import dump_config, read_config, validate_train_config
from rinchen.train.eole_run import CHECKPOINT_FILE
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.fake_ablation import (
    ID_SRC,
    ID_TGT,
    AblationRun,
    make_matrix,
    one_variant_axes,
    toy_plan,
    write_toy_corpus,
)
from tests.support.tiny_parallel import train_toy_spm

# (name, rows, weight, probability, upsampling) of the default variant, in the order
# of the plan: directions sorted, `auth` before `bt`.
DEFAULT_CORPORA = [
    (f"{EN_MN}.auth", 1200, 22, 0.225403, 0.88),
    (f"{EN_MN}.bt", 300, 6, 0.056351, 0.96),
    (f"{MN_EN}.auth", 1200, 22, 0.225403, 0.88),
    (f"{MN_EN}.bt", 300, 6, 0.056351, 0.96),
    (f"{MN_RU}.auth", 600, 15, 0.145497, 1.2),
    (f"{MN_RU}.bt", 300, 7, 0.072749, 1.12),
    (f"{RU_MN}.auth", 600, 15, 0.145497, 1.2),
    (f"{RU_MN}.bt", 300, 7, 0.072749, 1.12),
]


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


# --- prepare_variant -----------------------------------------------------------------


def _prepared(
    plan: AblationPlan, tmp_path: Path, index: int = 0, *, threads: int = 2
) -> PreparedVariant:
    return prepare_variant(
        _plan_in(plan, tmp_path), plan.variants[index], threads=threads
    )


def test_a_prepared_variant_lays_its_files_out_under_variants_and_its_id(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    directory = plan.out_dir / "variants" / ID_SRC
    assert prepared.planned is plan.variants[0]
    assert prepared.directory == directory
    assert prepared.config_path == directory / "config.yaml"
    assert prepared.run_dir == directory / "run"
    assert prepared.eval_dir == directory / "eval"
    assert prepared.result_path == directory / "result.json"
    assert prepared.ckpt_dir == directory / "run" / "ckpt"
    assert prepared.checkpoint_path == directory / "run" / "ckpt" / CHECKPOINT_FILE
    assert prepared.valid.src == directory / "valid.src"
    assert prepared.valid.tgt == directory / "valid.tgt"
    assert prepared.config_path.is_file()
    assert prepared.valid.src.is_file() and prepared.valid.tgt.is_file()
    # Nothing of the run itself exists yet.
    assert not prepared.run_dir.exists()
    assert not prepared.eval_dir.exists()
    assert not prepared.result_path.exists()


def test_a_prepared_variant_builds_the_corpora_and_names_the_weights_of_the_plan(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    assert [(c.name, c.rows) for c in prepared.corpora.corpora] == [
        (name, rows) for name, rows, *_ in DEFAULT_CORPORA
    ]
    assert prepared.weights == plan.variants[0].weights
    assert prepared.weights[f"{EN_MN}.auth"] == 22
    assert all(c.src_path.is_file() for c in prepared.corpora.corpora)
    assert (plan.out_dir / "data" / "auth" / f"{EN_MN}.auth.src").is_file()
    assert prepared.corpora.bt_by_direction == plan.variants[0].bt_by_direction


def test_the_rendered_config_passes_the_schema_and_carries_the_per_run_values(
    completed: Run,
) -> None:
    cfg = completed.config(ID_SRC)
    run_dir = completed.variant_dir(ID_SRC) / "run"
    validate_train_config(cfg)

    assert cfg["seed"] == 1234
    assert cfg["src_vocab"] == (run_dir / "vocab.src").as_posix()
    assert cfg["training"]["model_path"] == (run_dir / "ckpt").as_posix()
    spm = completed.plan.spm_model.as_posix()
    assert cfg["transforms_configs"]["sentencepiece"] == {
        "src_subword_model": spm,
        "tgt_subword_model": spm,
    }
    # The base's own values stay.
    assert cfg["training"]["batch_size"] == 2048
    assert cfg["model"]["hidden_size"] == 64
    assert cfg["transforms_configs"]["filtertoolong"] == {
        "src_seq_length": 64,
        "tgt_seq_length": 64,
    }
    assert cfg["overwrite"] is True and cfg["share_vocab"] is True


def test_the_rendered_config_has_the_weights_of_the_plan_and_the_prepared_corpus_files(
    completed: Run,
) -> None:
    cfg = completed.config(ID_SRC)
    data_dir = completed.plan.out_dir / "data"

    assert list(cfg["data"]) == [name for name, *_ in DEFAULT_CORPORA] + ["valid"]
    for name, _, weight, *_ in DEFAULT_CORPORA:
        corpus = cfg["data"][name]
        assert corpus["weight"] == weight
        group = name.rpartition(".")[2]
        folder = (
            data_dir / "auth" if group == "auth" else data_dir / "bt" / "r0.5_s1234"
        )
        assert corpus["path_src"] == (folder / f"{name}.src").as_posix()
        assert corpus["path_tgt"] == (folder / f"{name}.tgt").as_posix()
        assert corpus["transforms"] == ["sentencepiece", "prefix", "filtertoolong"]
    valid = cfg["data"]["valid"]
    directory = completed.variant_dir(ID_SRC)
    assert valid == {
        "path_src": (directory / "valid.src").as_posix(),
        "path_tgt": (directory / "valid.tgt").as_posix(),
    }


def test_the_tag_goes_to_the_source_prefix_for_src_and_to_the_target_prefix_for_tgt(
    completed: Run,
) -> None:
    src = completed.config(ID_SRC)["data"]
    tgt = completed.config(ID_TGT)["data"]

    assert (src[f"{EN_MN}.auth"]["src_prefix"], src[f"{EN_MN}.auth"]["tgt_prefix"]) == (
        "<2khk_Cyrl>",
        "",
    )
    assert src[f"{MN_EN}.bt"]["src_prefix"] == "<2eng_Latn> <bt>"
    assert src[f"{MN_EN}.bt"]["tgt_prefix"] == ""
    assert (tgt[f"{EN_MN}.auth"]["src_prefix"], tgt[f"{EN_MN}.auth"]["tgt_prefix"]) == (
        "",
        "<2khk_Cyrl>",
    )
    assert (tgt[f"{MN_RU}.bt"]["src_prefix"], tgt[f"{MN_RU}.bt"]["tgt_prefix"]) == (
        "<bt>",
        "<2rus_Cyrl>",
    )


def test_the_validation_files_hold_the_tag_where_the_placement_says(
    completed: Run,
) -> None:
    src_dir = completed.variant_dir(ID_SRC)
    tgt_dir = completed.variant_dir(ID_TGT)

    first_src = (src_dir / "valid.src").read_text(encoding="utf-8").split("\n")[0]
    first_tgt = (tgt_dir / "valid.tgt").read_text(encoding="utf-8").split("\n")[0]
    assert first_src.startswith("<2khk_Cyrl> ")
    assert first_tgt.startswith("<2khk_Cyrl> ")
    assert "<2khk_Cyrl>" not in (tgt_dir / "valid.src").read_text(encoding="utf-8")
    assert len((src_dir / "valid.src").read_text(encoding="utf-8").split("\n")) == 401


def test_every_path_in_the_config_is_absolute_posix_text(completed: Run) -> None:
    cfg = completed.config(ID_TGT)
    paths = [
        cfg["src_vocab"],
        cfg["training"]["model_path"],
        *cfg["transforms_configs"]["sentencepiece"].values(),
        *(
            corpus[key]
            for corpus in cfg["data"].values()
            for key in ("path_src", "path_tgt")
        ),
    ]

    assert all(Path(path).is_absolute() and "\\" not in path for path in paths)


def test_the_seed_of_the_variant_is_the_seed_key_of_the_config(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    plan = toy_plan(
        tmp_path, toy_spm, corpus=corpus, axes=one_variant_axes(), seeds=[7, 42]
    )

    second = prepare_variant(plan, plan.variants[1], threads=2)

    assert read_config(second.config_path, "config")["seed"] == 42
    assert second.planned.variant.id == "t0.5_bt0.5_src_s42"


def test_the_matrix_overrides_are_merged_over_the_base_and_keep_the_per_run_values(
    corpus: Path, toy_spm: Path, tmp_path: Path
) -> None:
    overrides = {
        "training": {"dropout": [0.25], "learning_rate": 0.7, "keep_checkpoint": None},
        "model": {"encoder": {"layers": 3}},
    }
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        overrides=overrides,
    )

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    cfg = read_config(prepared.config_path, "config")
    assert cfg["training"]["dropout"] == [0.25]
    assert cfg["training"]["learning_rate"] == 0.7
    # None deletes the base's key, though it merges into a section the run also sets.
    assert "keep_checkpoint" not in cfg["training"]
    assert cfg["training"]["model_path"] == (prepared.run_dir / "ckpt").as_posix()
    assert cfg["training"]["batch_size"] == 2048
    assert cfg["model"]["encoder"] == {"layers": 3}
    assert cfg["model"]["decoder"] == {"layers": 2}


def _base_without_filtertoolong(tmp_path: Path) -> Path:
    base = read_config(make_matrix().base, "base")
    del base["transforms_configs"]["filtertoolong"]
    path = tmp_path / "base_without.yaml"
    dump_config(base, path)
    return path


@pytest.mark.parametrize(
    ("base_has", "overrides", "expected"),
    [
        (True, {}, True),
        (False, {}, False),
        # The corpora use the transform when the finished config configures it.
        (
            False,
            {"transforms_configs": {"filtertoolong": {"src_seq_length": 32}}},
            True,
        ),
        (True, {"transforms_configs": {"filtertoolong": None}}, False),
    ],
)
def test_the_corpora_run_filtertoolong_when_the_finished_config_configures_it(
    corpus: Path,
    toy_spm: Path,
    tmp_path: Path,
    base_has: bool,
    overrides: dict[str, Any],
    expected: bool,
) -> None:
    base = make_matrix().base if base_has else _base_without_filtertoolong(tmp_path)
    plan = toy_plan(
        tmp_path,
        toy_spm,
        corpus=corpus,
        axes=one_variant_axes(),
        base=str(base),
        overrides=overrides,
    )

    prepared = prepare_variant(plan, plan.variants[0], threads=2)

    cfg = read_config(prepared.config_path, "config")
    transforms = {
        tuple(c["transforms"]) for n, c in cfg["data"].items() if n != "valid"
    }
    expected_transforms = ("sentencepiece", "prefix") + (
        ("filtertoolong",) if expected else ()
    )
    assert transforms == {expected_transforms}


def test_preparing_twice_gives_the_same_config_and_fingerprint_and_reuses_the_files(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    first = prepare_variant(plan, plan.variants[0], threads=2)
    second = prepare_variant(plan, plan.variants[0], threads=2)

    assert second.config_text == first.config_text
    assert second.fingerprint == first.fingerprint
    assert second.fingerprint_inputs == first.fingerprint_inputs
    assert not any(c.reused for c in first.corpora.corpora)
    assert all(c.reused for c in second.corpora.corpora)


def test_the_config_text_is_what_the_file_holds(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    prepared = _prepared(default_plan, tmp_path)

    assert prepared.config_text == prepared.config_path.read_text(encoding="utf-8")
    assert prepared.config_text.endswith("\n")


def _sha256_of_lines(lines: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _versions_with(**changed: str) -> Callable[[Sequence[str]], dict[str, str]]:
    """A stand-in for `tool_versions`: the version `1.0` of every package asked for,
    except those in `changed`."""
    return lambda packages: {name: changed.get(name, "1.0") for name in packages}


def test_the_fingerprint_inputs_name_every_input_by_its_digest(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    prepared = _prepared(default_plan, tmp_path, 1, threads=3)

    corpora = {
        c.name: [_sha256(c.src_path), _sha256(c.tgt_path)]
        for c in prepared.corpora.corpora
    }
    assert len(corpora) == 8
    assert prepared.fingerprint_inputs == {
        "config_sha256": hashlib.sha256(prepared.config_text.encode()).hexdigest(),
        "corpora": corpora,
        "validation": [_sha256(prepared.valid.src), _sha256(prepared.valid.tgt)],
        "spm_sha256": _sha256(default_plan.spm_model),
        "threads": 3,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("eole", "torch", "sentencepiece", "sacrebleu")
        },
        "dev_raw": {
            direction: [
                _sha256_of_lines(dev_set.src),
                _sha256_of_lines(dev_set.tgt),
            ]
            for direction, dev_set in default_plan.dev.sets.items()
        },
        "placement": "tgt",
    }
    canonical = json.dumps(
        prepared.fingerprint_inputs, sort_keys=True, separators=(",", ":")
    )
    assert prepared.fingerprint == hashlib.sha256(canonical.encode()).hexdigest()


def test_the_fingerprint_changes_with_each_of_its_inputs(
    default_plan: AblationPlan,
    tmp_path: Path,
    corpus: Path,
    toy_spm: Path,
    other_spm: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _prepared(default_plan, tmp_path)
    seen = {base.fingerprint}

    def distinct(fingerprint: str) -> None:
        assert fingerprint not in seen
        seen.add(fingerprint)

    # The thread count.
    distinct(_prepared(default_plan, tmp_path, threads=3).fingerprint)
    # The placement.
    distinct(_prepared(default_plan, tmp_path, 1).fingerprint)
    # The installed Eole.
    with monkeypatch.context() as patch:
        patch.setattr(ablation_prepare, "tool_versions", _versions_with(eole="9.9.9"))
        distinct(_prepared(default_plan, tmp_path).fingerprint)
    # The SentencePiece model, at another path and with other bytes at the same path.
    elsewhere = _plan_in(default_plan, tmp_path)
    distinct(
        prepare_variant(
            dataclasses.replace(elsewhere, spm_model=other_spm),
            elsewhere.variants[0],
            threads=2,
        ).fingerprint
    )
    in_place = tmp_path / "in_place.model"
    shutil.copy(toy_spm, in_place)
    plan = dataclasses.replace(elsewhere, spm_model=in_place)
    first = prepare_variant(plan, plan.variants[0], threads=2).fingerprint
    shutil.copy(other_spm, in_place)
    assert prepare_variant(plan, plan.variants[0], threads=2).fingerprint != first
    # An override.
    other = toy_plan(
        tmp_path / "ov",
        toy_spm,
        corpus=corpus,
        overrides={"training": {"dropout": [0.1]}},
    )
    distinct(
        prepare_variant(
            dataclasses.replace(other, out_dir=elsewhere.out_dir),
            other.variants[0],
            threads=2,
        ).fingerprint
    )


@pytest.mark.parametrize("package", ["eole", "torch", "sentencepiece", "sacrebleu"])
def test_the_version_of_each_package_that_shapes_a_model_or_a_score_is_a_fingerprint_input(
    default_plan: AblationPlan,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package: str,
) -> None:
    # Eole trains and loads the model, torch runs it, SentencePiece cuts the text and
    # sacrebleu scores it: another version of any of them is another model or score.
    with monkeypatch.context() as patch:
        patch.setattr(ablation_prepare, "tool_versions", _versions_with())
        before = _prepared(default_plan, tmp_path)
        patch.setattr(
            ablation_prepare, "tool_versions", _versions_with(**{package: "2.0"})
        )
        after = _prepared(default_plan, tmp_path)

    assert before.fingerprint_inputs["versions"] == dict.fromkeys(
        ["eole", "torch", "sentencepiece", "sacrebleu"], "1.0"
    )
    assert after.fingerprint_inputs["versions"][package] == "2.0"
    assert after.fingerprint != before.fingerprint


def test_a_package_that_is_not_installed_is_recorded_as_such_not_an_error(
    default_plan: AblationPlan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_version = importlib.metadata.version

    def version(name: str) -> str:
        if name == "sacrebleu":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", version)

    prepared = _prepared(default_plan, tmp_path)

    assert prepared.fingerprint_inputs["versions"]["sacrebleu"] == "not installed"
    assert prepared.fingerprint_inputs["versions"]["eole"] == real_version("eole")


@pytest.mark.parametrize("side", ["src", "tgt"])
def test_a_change_in_the_raw_dev_text_is_a_change_of_fingerprint_though_the_tokens_stay(
    default_plan: AblationPlan, tmp_path: Path, side: str
) -> None:
    # SentencePiece drops a trailing space, so the validation file is the same bytes,
    # but the references the chrF is computed against (and the source text the model
    # is asked to translate) are not.
    direction = EN_MN
    original = default_plan.dev.sets[direction]
    changed_set = dataclasses.replace(
        original,
        **{side: tuple(f"{line} " for line in getattr(original, side))},
    )
    changed = dataclasses.replace(
        default_plan,
        dev=dataclasses.replace(
            default_plan.dev, sets={**default_plan.dev.sets, direction: changed_set}
        ),
    )

    before = _prepared(default_plan, tmp_path / "before")
    after = _prepared(changed, tmp_path / "after")

    assert after.valid.src.read_bytes() == before.valid.src.read_bytes()
    assert after.valid.tgt.read_bytes() == before.valid.tgt.read_bytes()
    assert (
        after.fingerprint_inputs["validation"]
        == before.fingerprint_inputs["validation"]
    )
    assert (
        after.fingerprint_inputs["dev_raw"][direction]
        != before.fingerprint_inputs["dev_raw"][direction]
    )
    assert after.fingerprint != before.fingerprint


def test_a_dev_text_that_moves_to_another_line_is_another_fingerprint(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    # The hash is of the lines in order, each ended by an LF: swapping two lines of a
    # direction (same set of texts) must not give the same digest.
    direction = EN_MN
    original = default_plan.dev.sets[direction]
    swapped = dataclasses.replace(
        original,
        tgt=(original.tgt[1], original.tgt[0], *original.tgt[2:]),
    )
    changed = dataclasses.replace(
        default_plan,
        dev=dataclasses.replace(
            default_plan.dev, sets={**default_plan.dev.sets, direction: swapped}
        ),
    )

    before = _prepared(default_plan, tmp_path / "before")
    after = _prepared(changed, tmp_path / "after")

    assert (
        after.fingerprint_inputs["dev_raw"][direction][1]
        != before.fingerprint_inputs["dev_raw"][direction][1]
    )


def test_a_line_boundary_moved_inside_the_raw_dev_text_is_another_fingerprint(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    # The same characters in the same order, cut into lines elsewhere.
    direction = EN_MN
    original = default_plan.dev.sets[direction]
    first, second, *rest = original.tgt
    moved = dataclasses.replace(original, tgt=(first + second[:1], second[1:], *rest))
    changed = dataclasses.replace(
        default_plan,
        dev=dataclasses.replace(
            default_plan.dev, sets={**default_plan.dev.sets, direction: moved}
        ),
    )

    before = _prepared(default_plan, tmp_path / "before")
    after = _prepared(changed, tmp_path / "after")

    assert (
        after.fingerprint_inputs["dev_raw"][direction][1]
        != before.fingerprint_inputs["dev_raw"][direction][1]
    )


def test_moving_the_output_directory_changes_the_fingerprint_through_the_config_paths(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    # The rendered config names absolute paths, so a relocated run is another run (it
    # trains again); the checkpoint the manifest hashes is the artefact of record.
    first = _prepared(default_plan, tmp_path / "one")
    second = _prepared(default_plan, tmp_path / "two")

    assert first.fingerprint != second.fingerprint
    assert (
        first.fingerprint_inputs["config_sha256"]
        != second.fingerprint_inputs["config_sha256"]
    )


def test_the_threads_must_be_at_least_one(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = _plan_in(default_plan, tmp_path)

    with pytest.raises(ValueError, match="threads"):
        prepare_variant(plan, plan.variants[0], threads=0)

    assert not plan.out_dir.exists()


def test_one_thread_is_enough(default_plan: AblationPlan, tmp_path: Path) -> None:
    prepared = _prepared(default_plan, tmp_path, threads=1)

    assert prepared.fingerprint_inputs["threads"] == 1
    plan = _plan_in(default_plan, tmp_path)
    run = Run(plan, threads=1)
    assert {call.threads for call in run.train.calls} == {1}


def test_a_missing_sentencepiece_model_is_a_config_error_at_preparation(
    default_plan: AblationPlan, tmp_path: Path
) -> None:
    plan = dataclasses.replace(
        _plan_in(default_plan, tmp_path), spm_model=tmp_path / "gone.model"
    )

    with pytest.raises(ConfigError, match="gone.model"):
        prepare_variant(plan, plan.variants[0], threads=2)
