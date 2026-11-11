"""The shipped profile of the tiny pipeline, `configs/tiny/tiny.yaml`, its Eole base config
`configs/eole/tiny.yaml` and its registry `configs/data/sources.tiny_parallel.yaml` (D15, spec P5,
P7, P8, P12 and P14).

The profile is read from the file's bytes and pinned literal by literal: it is a table of priors from
one machine's measurements, and a change of one of its numbers is a deliberate edit of this test. A
threshold is never lowered (a miss is answered with more steps in `train.steps`). The files the
profile names must exist, the fixtures' pinned digests must be the digests of the committed files, and
every section must be accepted by the closed configuration of the stage it feeds, so that a tiny run
cannot start with a profile that a stage's own loader would refuse.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, get_args

import pytest
import yaml

from rinchen.acquire.registry import LocalSourceEntry, fetch_source, load_registry
from rinchen.config import tiny as tiny_module
from rinchen.config.ablation import AblationMatrix
from rinchen.config.filters import (
    Bounds,
    FilterConfig,
    LabseConfig,
    LengthConfig,
    MinHashConfig,
    QeConfig,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.config.serve import load_serve_config
from rinchen.config.spm import SpmConfig
from rinchen.config.tiny import (
    STAGE_IDS,
    ThresholdsSection,
    TinyConfig,
    config_path_for,
    load_tiny_config,
)
from rinchen.eval.bootstrap import MIN_RESAMPLES
from rinchen.eval.suite_rules import SPLITS
from rinchen.tiny.fixtures import FIXTURE_FILES, fixture_digests
from rinchen.train.eole_cfg import validate_train_config
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_scans import hits

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED: Final = REPO_ROOT / "configs" / "tiny" / "tiny.yaml"
EOLE_BASE: Final = REPO_ROOT / "configs" / "eole" / "tiny.yaml"
TEACHER_BASE: Final = REPO_ROOT / "configs" / "eole" / "teacher_tiny.yaml"
REGISTRY: Final = REPO_ROOT / "configs" / "data" / "sources.tiny_parallel.yaml"
FIXTURES: Final = REPO_ROOT / "tests" / "fixtures" / "parallel"
THE_FILES_OF_THIS_TASK: Final = (
    SHIPPED,
    EOLE_BASE,
    REGISTRY,
    REPO_ROOT / "src" / "rinchen" / "config" / "tiny.py",
    REPO_ROOT / "src" / "rinchen" / "tiny" / "states.py",
    REPO_ROOT / "src" / "rinchen" / "tiny" / "context.py",
    REPO_ROOT / "src" / "rinchen" / "tiny" / "thresholds.py",
)

# The digests of the committed fixtures, as the profile pins them: a changed fixture is a
# re-measurement of the spec's P5, P7 and P8, and this table moves with it, on purpose.
PINNED_DIGESTS: Final = {
    "eng_Latn-khk_Cyrl.tsv": "7d1d15d7b7fc8e6ed5bc617990505519f37909ea0107f221728fdb38a962d229",
    "rus_Cyrl-khk_Cyrl.tsv": "dce119b354e907812361dce126b3d1266bebe874fc594c89a6247d3a0a8f6bbe",
    "mn_Mong.tsv": "16000ecdd3c099179c5e243cfa2a9c6d89bba565091cf680e7ac3231f0ad0f34",
    "mono_khk_Cyrl.tsv": "79f488e2ffe757dd34194b0ef056e41f29eb038c8b2d84dde57e23e7fa418952",
    "noise.tsv": "fa16dba1968d182ff50fc7f61c62b86b16fd7112481196c17ae70dfacafe3248",
    "nouns.tsv": "62fe9458a7d8592cb61ecedcaf2b1023b4a130fb357cb28b4e373d54fa2387be",
    "entities.tsv": "6cb6bb4ec68a042ff3a721d20798853d633fc4d36d1d2ed5e36e772f755cef63",
}
TIMEOUTS: Final = {
    "environment": 30,
    "fetch": 30,
    "corpus": 60,
    "filter": 120,
    "spm": 120,
    "converter": 300,
    "synth": 120,
    "prepare": 120,
    "train": 600,
    "average": 60,
    "export": 120,
    "parity": 180,
    "serve": 60,
    "robustness": 120,
    "evaluate": 300,
}
# What a device value or key would be called, in pieces: the scans of the gate read the tests of the
# pipeline as well and no file of it may hold these words.
DEVICE_WORDS: Final = {"dev" + "ice", "cu" + "da", "au" + "to", "gpu", "gpus"}


def shipped() -> TinyConfig:
    return load_tiny_config(SHIPPED.read_bytes())


def raw_document(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---- the profile ---------------------------------------------------------------------------------


def test_the_shipped_profile_loads_and_holds_the_values_of_the_spec() -> None:
    config = shipped()
    assert config.train.steps == 1500
    assert config.train.average_last == 2
    assert config.train.temperature == 1.0
    assert config.train.bt_ratio == 0.2
    assert config.train.tag_placement == "src"
    assert config.train.dev_max_lines == 100
    assert config.decode.beam_size == 1
    assert (config.spm.vocab_size, config.spm.input_sentence_size) == (640, 100000)
    assert (config.spm.character_coverage, config.spm.temperature) == (0.9999, 0.3)
    assert config.filter.char_ratio_min == 0.4
    assert config.filter.char_ratio_max == 2.5
    assert config.filter.minhash_threshold == 0.8
    assert config.filter.labse_min_cosine == 0.7
    assert config.filter.qe_max_metricx == 8.0
    assert config.filter.max_authentic_flagged_share == 0.01
    assert config.converter.char_vocab_size == 200
    assert (config.synth.job, config.synth.shard_size, config.synth.n) == (
        "bt_eng_khk",
        32,
        1,
    )
    assert config.export.quantizations == ("float32", "int8")
    assert config.parity.max_lines == 50
    assert config.evaluation.memorised_split == "test"
    assert config.evaluation.heldout_split == "devtest"
    assert config.evaluation.resamples == 1000
    assert config.budget_seconds == 1200
    assert config.threads == 2
    assert config.qe_fake is True
    assert config.created == "2026-11-11T00:00:00Z"


def test_the_shipped_thresholds_are_the_master_plans_and_never_lower() -> None:
    thresholds = shipped().thresholds
    assert thresholds.model_dump() == {
        "train_chrf_min": 50.0,
        "parity_exact_min": 0.95,
        "parity_delta_max": 0.001,
        "int8_chrf_drop_max": 1.0,
        "script_purity_min": 1.0,
        "placeholder_acc_min": 1.0,
    }
    assert thresholds == ThresholdsSection()


def test_the_timeouts_are_the_fifteen_of_the_spec_each_at_least_30_seconds() -> None:
    timeouts = shipped().timeouts
    assert dict(timeouts) == TIMEOUTS
    assert tuple(timeouts) == STAGE_IDS
    assert min(timeouts.values()) >= 30


def test_the_stamp_is_a_utc_time() -> None:
    parsed = datetime.strptime(shipped().created, "%Y-%m-%dT%H:%M:%SZ")
    assert parsed.replace(tzinfo=UTC).isoformat() == "2026-11-11T00:00:00+00:00"


def test_the_profile_says_in_its_header_that_every_number_is_a_prior() -> None:
    lines = SHIPPED.read_text(encoding="utf-8").splitlines()
    header = []
    for line in lines:
        if not line.startswith("#"):
            break
        header.append(line)
    assert header
    text = " ".join(" ".join(header).lower().split())
    assert "every number below is a prior" in text


def test_the_profile_is_plain_ascii_text_with_one_final_line_feed() -> None:
    for path in (SHIPPED, EOLE_BASE, REGISTRY):
        data = path.read_bytes()
        assert data.isascii(), path.name
        assert data.endswith(b"\n") and not data.endswith(b"\n\n"), path.name
        assert b"\t" not in data and b"\r" not in data, path.name


def test_the_profile_holds_no_device_key_and_no_device_value() -> None:
    def words(node: Any) -> list[str]:
        if isinstance(node, dict):
            return [
                word
                for key, value in node.items()
                for word in [*re.findall(r"[a-z]+", str(key).lower()), *words(value)]
            ]
        if isinstance(node, list):
            return [word for item in node for word in words(item)]
        return re.findall(r"[a-z]+", str(node).lower())

    assert DEVICE_WORDS.isdisjoint(words(raw_document(SHIPPED)))


def test_the_shipped_path_of_a_profile_is_the_one_the_command_reads() -> None:
    assert config_path_for("tiny") == SHIPPED
    assert SHIPPED.is_file()


def test_no_full_profile_is_shipped() -> None:
    # A table of unmeasured priors that a command would run is how a 14-day job starts by mistake.
    assert not config_path_for("full").exists()
    assert sorted(path.name for path in SHIPPED.parent.iterdir()) == ["tiny.yaml"]


# ---- the files the profile names -----------------------------------------------------------------


def test_every_path_of_the_profile_exists_under_the_repository() -> None:
    config = shipped()
    files = [
        config.registry,
        config.converter.pairs,
        config.train.base,
        config.serve.config,
    ]
    for path in files:
        assert path.is_file(), path.name
        assert path.is_relative_to(REPO_ROOT), path.name
    assert config.fixtures.directory.is_dir()
    assert config.fixtures.directory == FIXTURES
    assert config.train.base == EOLE_BASE
    assert config.registry == REGISTRY
    assert config.serve.config == REPO_ROOT / "configs" / "serve" / "tiny.yaml"
    assert (
        config.converter.pairs
        == REPO_ROOT / "tests" / "fixtures" / "script_conv" / "tiny_words.tsv"
    )


def test_the_pinned_digests_are_those_of_the_committed_fixtures() -> None:
    pinned = shipped().fixtures.sha256
    assert tuple(pinned) == FIXTURE_FILES
    assert dict(pinned) == fixture_digests(FIXTURES)


def test_the_pinned_digests_are_the_literals_of_the_ledger() -> None:
    assert dict(shipped().fixtures.sha256) == PINNED_DIGESTS
    assert tuple(PINNED_DIGESTS) == FIXTURE_FILES


# ---- the registry -----------------------------------------------------------------------------


def test_the_registry_loads_and_names_the_seven_fixtures_as_local_sources() -> None:
    entries = load_registry(shipped().registry)
    assert len(entries) == 7
    assert all(isinstance(entry, LocalSourceEntry) for entry in entries)
    urls = sorted(entry.url for entry in entries if isinstance(entry, LocalSourceEntry))
    assert urls == sorted(
        f"local://tests/fixtures/parallel/{name}" for name in FIXTURE_FILES
    )
    ids = [entry.id for entry in entries]
    assert len(set(ids)) == 7
    assert all(entry.expected_sha is None for entry in entries)


def test_fetching_the_registry_copies_every_fixture_byte_for_byte(
    tmp_path: Path,
) -> None:
    for entry in load_registry(shipped().registry):
        copied = fetch_source(entry, tmp_path)
        assert copied.read_bytes() == (FIXTURES / copied.name).read_bytes()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(FIXTURE_FILES)


# ---- the Eole base config -----------------------------------------------------------------------


def test_the_eole_base_config_passes_the_validator_of_the_training_configs() -> None:
    validate_train_config(raw_document(EOLE_BASE))


def test_the_eole_base_config_is_the_tiny_model_every_model_of_the_project_must_be() -> (
    None
):
    cfg = raw_document(EOLE_BASE)
    assert (
        cfg["model"]["embeddings"]["position_encoding_type"] == "SinusoidalInterleaved"
    )
    assert cfg["model"]["architecture"] == "transformer"
    assert cfg["training"]["compute_dtype"] == "fp32"
    assert cfg["training"]["gpu_ranks"] == []
    assert cfg["training"]["world_size"] == 1
    assert cfg["data"] == {}
    assert cfg["share_vocab"] is True
    assert cfg["overwrite"] is True
    # Trained from random initialisation: nothing to start from.
    assert "train_from" not in cfg["training"]
    assert "reset_optim" not in cfg["training"]


def test_the_eole_base_config_holds_the_schedule_of_the_profile() -> None:
    cfg = raw_document(EOLE_BASE)
    training = cfg["training"]
    assert training["train_steps"] == 1500
    assert training["valid_steps"] == 1500
    assert training["save_checkpoint_steps"] == 100
    assert training["keep_checkpoint"] == 4
    assert cfg["report_every"] == 10
    assert training["train_steps"] == shipped().train.steps
    assert shipped().train.average_last <= training["keep_checkpoint"]


def test_the_eole_base_config_is_the_tiny_model_of_the_teacher_with_three_values_changed() -> (
    None
):
    base, teacher = raw_document(EOLE_BASE), raw_document(TEACHER_BASE)
    changed = {
        "train_steps": (1500, 200),
        "valid_steps": (1500, 50),
        "save_checkpoint_steps": (100, 50),
    }
    for key, (ours, theirs) in changed.items():
        assert base["training"][key] == ours
        assert teacher["training"][key] == theirs
        base["training"][key] = teacher["training"][key]
    assert base == teacher


def test_the_model_is_the_2_plus_2_layer_one_of_the_spec() -> None:
    model = raw_document(EOLE_BASE)["model"]
    assert model["hidden_size"] == 64
    assert model["encoder"] == {"layers": 2}
    assert model["decoder"] == {"layers": 2}
    assert model["heads"] == 4
    assert model["transformer_ff"] == 128
    assert model["embeddings"]["word_vec_size"] == 64
    training = raw_document(EOLE_BASE)["training"]
    assert (training["optim"], training["decay_method"]) == ("adamw", "noam")
    assert (training["learning_rate"], training["warmup_steps"]) == (0.5, 100)
    assert (training["batch_type"], training["batch_size"]) == ("tokens", 2048)
    assert training["dropout"] == [0.0]
    assert training["label_smoothing"] == 0.1
    assert (training["bucket_size"], training["num_workers"]) == (1024, 1)


# ---- what each section feeds accepts it ------------------------------------------------------


def test_the_spm_section_makes_a_valid_sentencepiece_config() -> None:
    section = shipped().spm
    config = SpmConfig.model_validate(
        {**section.model_dump(), "seed": 1234, "corpus_dir": "a", "output_prefix": "b"}
    )
    assert config.vocab_size == 640


def test_the_filter_section_makes_a_valid_filter_config() -> None:
    section = shipped().filter
    config = FilterConfig(
        stages=["script_ratio", "numbers", "length", "minhash", "labse", "qe"],
        length=LengthConfig(
            char_ratio=Bounds(min=section.char_ratio_min, max=section.char_ratio_max)
        ),
        minhash=MinHashConfig(threshold=section.minhash_threshold),
        labse=LabseConfig(min_cosine=section.labse_min_cosine, device="cpu"),
        qe=QeConfig(max_metricx_qe=section.qe_max_metricx),
    )
    assert config.length.char_ratio.max == 2.5


def test_the_train_section_makes_a_valid_ablation_matrix() -> None:
    section = shipped().train
    matrix = AblationMatrix.model_validate(
        {
            "name": "tiny",
            "base": str(section.base),
            "axes": {
                "tag_placement": [section.tag_placement],
                "temperature": [section.temperature],
                "bt_ratio": [section.bt_ratio],
            },
            "seeds": [1234],
            "dev_max_lines": section.dev_max_lines,
        }
    )
    assert matrix.dev_max_lines == 100


def test_the_serve_section_names_a_serve_config_that_loads() -> None:
    config = load_serve_config(shipped().serve.config, env={})
    assert config.device == "cpu"


def test_the_two_splits_and_the_resamples_are_those_the_suites_and_the_report_allow() -> (
    None
):
    assert get_args(tiny_module.Split) == SPLITS
    config = shipped()
    assert config.evaluation.memorised_split in SPLITS
    assert config.evaluation.heldout_split in SPLITS
    assert config.evaluation.resamples >= MIN_RESAMPLES


def test_the_profile_seed_is_in_the_profile_file_of_the_repository_and_not_here() -> (
    None
):
    assert "seed" not in raw_document(SHIPPED)
    assert (
        raw_document(REPO_ROOT / "configs" / "profiles" / "tiny.yaml")["seed"] == 1234
    )


# ---- the content rules -------------------------------------------------------------------------


@pytest.mark.parametrize("path", THE_FILES_OF_THIS_TASK, ids=lambda p: p.name)
def test_no_file_of_this_task_holds_a_word_the_project_excludes(path: Path) -> None:
    assert hits(path.read_text(encoding="utf-8")) == []
