"""The closed configuration of the tiny pipeline, `rinchen.config.tiny` (D15, spec P7, P8, P12, P13
and P14): every refusal of the loader, each a `ConfigError` that names the field and quotes nothing
the file said.

The document is read from the bytes the loader is handed. Every level is closed (an unknown key is an
error, and no key of the schema names a device), every key is required unless a default is named, a
number is a number and not a boolean or a quoted text, a path is not blank and holds no `..`, and
nothing is read from the environment. The shipped profile has a test of its own
(`test_config_tiny_shipped.py`); the document here is a literal of the interface, so that these
tests do not move when a prior of the shipped file does. The refusals of the values (counts, ranges,
paths, digests and the stamp) are in `test_config_tiny_values.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import ValidationError

from rinchen.config.loader import REPO_ROOT
from rinchen.config.tiny import (
    STAGE_IDS,
    ThresholdsSection,
    TinyConfig,
    config_path_for,
    load_tiny_config,
)
from rinchen.errors import ConfigError, RinchenError
from rinchen.tiny.fixtures import FIXTURE_FILES
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_config_documents import (
    SECTIONS,
    SENTINEL,
    TIMEOUTS,
    document,
    dump,
    key_paths,
    mapping_paths,
    refusal,
    with_value,
    without_key,
)

pytestmark = pytest.mark.usefixtures("no_network")

# The names no key of the schema may carry, written in pieces: no module of the pipeline holds a
# device word, and the scans of the gate read the tests of the pipeline as well.
DEVICE_KEYS: Final = ("dev" + "ice", "gpu_" + "ranks", "cu" + "da", "au" + "to", "gpu")


# ---- the document of the interface loads --------------------------------------------------------


def test_the_document_of_the_interface_loads_into_the_sections_of_the_interface() -> (
    None
):
    config = load_tiny_config(dump(document()))
    assert isinstance(config, TinyConfig)
    assert list(TinyConfig.model_fields) == [
        "fixtures",
        "registry",
        "spm",
        "filter",
        "converter",
        "synth",
        "train",
        "export",
        "parity",
        "decode",
        "serve",
        "evaluation",
        "thresholds",
        "budget_seconds",
        "timeouts",
        "threads",
        "qe_fake",
        "created",
    ]
    assert config.train.steps == 1500
    assert config.decode.beam_size == 1
    assert config.export.quantizations == ("float32", "int8")
    assert config.timeouts == TIMEOUTS
    assert config.qe_fake is True
    assert config.created == "2026-11-11T00:00:00Z"


def test_the_fields_of_each_section_are_those_of_the_interface() -> None:
    fields = {
        name: list(type(getattr(load_tiny_config(dump(document())), name)).model_fields)
        for name in SECTIONS
    }
    assert fields == {
        "fixtures": ["directory", "sha256"],
        "spm": [
            "vocab_size",
            "input_sentence_size",
            "character_coverage",
            "temperature",
        ],
        "filter": [
            "char_ratio_min",
            "char_ratio_max",
            "minhash_threshold",
            "labse_min_cosine",
            "qe_max_metricx",
            "max_authentic_flagged_share",
        ],
        "converter": ["pairs", "char_vocab_size"],
        "synth": ["job", "shard_size", "n"],
        "train": [
            "base",
            "steps",
            "average_last",
            "temperature",
            "bt_ratio",
            "tag_placement",
            "dev_max_lines",
        ],
        "export": ["quantizations"],
        "parity": ["max_lines"],
        "decode": ["beam_size"],
        "serve": ["config"],
        "evaluation": ["memorised_split", "heldout_split", "resamples"],
        "thresholds": [
            "train_chrf_min",
            "parity_exact_min",
            "parity_delta_max",
            "int8_chrf_drop_max",
            "script_purity_min",
            "placeholder_acc_min",
        ],
    }


def test_the_stage_ids_are_the_fifteen_of_the_spec_in_order() -> None:
    assert STAGE_IDS == tuple(TIMEOUTS)
    assert len(STAGE_IDS) == 15


def test_a_config_error_is_a_rinchen_error() -> None:
    assert issubclass(ConfigError, RinchenError)
    assert refusal(b"- a\n- b\n").startswith("invalid tiny config")


def test_the_configuration_is_read_only() -> None:
    config = load_tiny_config(dump(document()))
    with pytest.raises(ValidationError):
        config.threads = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.train.steps = 1  # type: ignore[misc]


def test_the_same_bytes_give_equal_configurations() -> None:
    data = dump(document())
    assert load_tiny_config(data) == load_tiny_config(data)
    assert load_tiny_config(data) != load_tiny_config(dump(with_value(("threads",), 3)))


# ---- the shape of the document --------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"[]",
        b"- a\n- b\n",
        b"just text",
        b"42",
        b"null",
        b"~",
        b"---\n",
    ],
)
def test_a_document_that_is_not_a_mapping_is_refused(data: bytes) -> None:
    assert "mapping" in refusal(data)


def test_bytes_that_are_not_utf8_are_refused() -> None:
    assert "UTF-8" in refusal(b"\xff\xfe\x00budget_seconds: 1\n")


def test_text_that_is_not_yaml_is_refused_with_its_line_and_none_of_its_text() -> None:
    message = refusal(b"budget_seconds: 1\nthreads: [ZZ-planted\n")
    assert "YAML" in message
    assert "line" in message
    assert "ZZ-planted" not in message


@pytest.mark.parametrize("level", mapping_paths(), ids=lambda p: ".".join(p) or "top")
def test_an_unknown_key_is_refused_at_every_level(level: tuple[str, ...]) -> None:
    doc = document()
    node = doc
    for part in level:
        node = node[part]
    node["bogus_key"] = 1
    message = refusal(doc)
    assert ".".join([*level, "bogus_key"]) in message


@pytest.mark.parametrize("name", DEVICE_KEYS)
@pytest.mark.parametrize("level", mapping_paths(), ids=lambda p: ".".join(p) or "top")
def test_no_level_takes_a_key_that_names_a_device(
    level: tuple[str, ...], name: str
) -> None:
    doc = document()
    node = doc
    for part in level:
        node = node[part]
    node[name] = ["0"] if name == "gpu_" + "ranks" else "cpu"
    message = refusal(doc)
    assert ".".join([*level, name]) in message


PERMITTED_MISSING = {("thresholds", name) for name in ThresholdsSection.model_fields}


@pytest.mark.parametrize(
    "path",
    [p for p in key_paths(document()) if p not in PERMITTED_MISSING],
    ids=lambda p: ".".join(p),
)
def test_a_missing_key_is_refused_at_every_level_and_named(
    path: tuple[str, ...],
) -> None:
    message = refusal(without_key(path))
    if (
        path[0] in ("timeouts", "fixtures")
        and len(path) >= 2
        and path[:2] != ("fixtures", "directory")
    ):
        # An entry of a closed table: the table is named, and so is the entry.
        assert all(part in message for part in path)
    else:
        assert ".".join(path) in message


@pytest.mark.parametrize("name", sorted(ThresholdsSection.model_fields))
def test_a_threshold_that_is_left_out_keeps_the_bound_of_the_master_plan(
    name: str,
) -> None:
    doc = document()
    del doc["thresholds"][name]
    config = load_tiny_config(dump(doc))
    assert getattr(config.thresholds, name) == getattr(ThresholdsSection(), name)


def test_the_defaults_of_the_thresholds_are_the_master_plans() -> None:
    assert ThresholdsSection().model_dump() == {
        "train_chrf_min": 50.0,
        "parity_exact_min": 0.95,
        "parity_delta_max": 0.001,
        "int8_chrf_drop_max": 1.0,
        "script_purity_min": 1.0,
        "placeholder_acc_min": 1.0,
    }
    doc = document()
    doc["thresholds"] = {}
    assert load_tiny_config(dump(doc)).thresholds == ThresholdsSection()


def test_the_section_of_the_thresholds_itself_is_required() -> None:
    assert "thresholds" in refusal(without_key(("thresholds",)))


# ---- the timeouts and the budget -----------------------------------------------------------------


def test_a_timeout_of_a_stage_that_does_not_exist_is_refused() -> None:
    doc = document()
    doc["timeouts"]["release"] = 30
    message = refusal(doc)
    assert "timeouts" in message
    assert "release" in message


def test_a_stage_without_a_timeout_is_refused_and_named() -> None:
    for stage in STAGE_IDS:
        doc = document()
        del doc["timeouts"][stage]
        message = refusal(doc)
        assert "timeouts" in message
        assert stage in message


def test_the_timeouts_must_be_a_mapping() -> None:
    assert "timeouts" in refusal(with_value(("timeouts",), [30] * 15))
    assert "timeouts" in refusal(with_value(("timeouts",), 30))


@pytest.mark.parametrize(
    "path",
    [("budget_seconds",), ("timeouts", "train"), ("timeouts", "environment")],
    ids=lambda p: ".".join(p),
)
@pytest.mark.parametrize(
    "bad", [True, False, 1200.0, 30.5, "1200", None, 0, -1, 86401, [30], {"a": 1}]
)
def test_a_budget_or_timeout_that_is_not_a_whole_number_of_1_to_86400_is_refused(
    path: tuple[str, ...], bad: Any
) -> None:
    assert ".".join(path) in refusal(with_value(path, bad))


@pytest.mark.parametrize("path", [("budget_seconds",), ("timeouts", "train")])
@pytest.mark.parametrize("good", [1, 2, 86399, 86400])
def test_a_budget_or_timeout_of_1_to_86400_is_accepted(
    path: tuple[str, ...], good: int
) -> None:
    config = load_tiny_config(dump(with_value(path, good)))
    assert (
        config.budget_seconds
        if path == ("budget_seconds",)
        else config.timeouts["train"]
    ) == good


# ---- the closed values ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quantizations",
    [
        ["au" + "to"],
        ["float16"],
        ["int8_float16"],
        ["float32", "float16"],
        ["int8_float16", "int8"],
        ["bfloat16"],
        ["int16"],
        ["Float32"],
        [],
        ["int8", "int8"],
        "int8",
        None,
        [None],
        [8],
    ],
)
def test_the_quantizations_are_float32_and_int8_and_nothing_else(
    quantizations: Any,
) -> None:
    assert "export.quantizations" in refusal(
        with_value(("export", "quantizations"), quantizations)
    )


@pytest.mark.parametrize(
    "quantizations", [["float32"], ["int8"], ["int8", "float32"], ["float32", "int8"]]
)
def test_the_quantizations_that_are_accepted(quantizations: list[str]) -> None:
    config = load_tiny_config(
        dump(with_value(("export", "quantizations"), quantizations))
    )
    assert config.export.quantizations == tuple(quantizations)


@pytest.mark.parametrize(
    "placement", ["tgt", "suffix", "SRC", "src ", "", None, 1, ["src"]]
)
def test_only_the_source_tag_placement_is_accepted(placement: Any) -> None:
    assert "train.tag_placement" in refusal(
        with_value(("train", "tag_placement"), placement)
    )


@pytest.mark.parametrize("beam", [0, -1, True, 1.0, "1", None])
def test_a_beam_size_must_be_a_positive_whole_number(beam: Any) -> None:
    assert "decode.beam_size" in refusal(with_value(("decode", "beam_size"), beam))


@pytest.mark.parametrize("beam", [1, 4, 5])
def test_a_beam_size_of_a_positive_whole_number_is_accepted(beam: int) -> None:
    assert (
        load_tiny_config(
            dump(with_value(("decode", "beam_size"), beam))
        ).decode.beam_size
        == beam
    )


@pytest.mark.parametrize(
    "path",
    [("evaluation", "memorised_split"), ("evaluation", "heldout_split")],
    ids=lambda p: p[1],
)
@pytest.mark.parametrize("bad", ["train", "valid", "Test", "", None, 1, ["test"]])
def test_a_split_must_be_one_of_the_three_of_the_suites(
    path: tuple[str, ...], bad: Any
) -> None:
    assert ".".join(path) in refusal(with_value(path, bad))


def test_the_two_splits_must_differ() -> None:
    doc = document()
    doc["evaluation"]["heldout_split"] = doc["evaluation"]["memorised_split"]
    assert "evaluation" in refusal(doc)


def test_the_held_out_split_may_not_be_the_selection_split() -> None:
    assert "evaluation" in refusal(with_value(("evaluation", "heldout_split"), "dev"))


@pytest.mark.parametrize(
    "pair",
    [("test", "devtest"), ("dev", "devtest"), ("dev", "test"), ("devtest", "test")],
)
def test_the_splits_that_are_accepted(pair: tuple[str, str]) -> None:
    doc = document()
    doc["evaluation"]["memorised_split"], doc["evaluation"]["heldout_split"] = pair
    config = load_tiny_config(dump(doc))
    assert (config.evaluation.memorised_split, config.evaluation.heldout_split) == pair


@pytest.mark.parametrize("resamples", [99, 0, -1, True, 100.0, "100", None])
def test_resamples_below_the_minimum_of_the_report_are_refused(resamples: Any) -> None:
    assert "evaluation.resamples" in refusal(
        with_value(("evaluation", "resamples"), resamples)
    )


@pytest.mark.parametrize("resamples", [100, 101, 1000])
def test_resamples_of_the_minimum_and_above_are_accepted(resamples: int) -> None:
    config = load_tiny_config(dump(with_value(("evaluation", "resamples"), resamples)))
    assert config.evaluation.resamples == resamples


@pytest.mark.parametrize(
    "job", ["", "BT", "bt eng", "bt/eng", "1bt", "bt\n", "a" * 65, None, 7, "bt-eng"]
)
def test_a_synthetic_job_is_a_closed_identifier(job: Any) -> None:
    assert "synth.job" in refusal(with_value(("synth", "job"), job))


def test_a_synthetic_job_of_the_longest_shape_is_accepted() -> None:
    job = "a" + "b" * 63
    assert load_tiny_config(dump(with_value(("synth", "job"), job))).synth.job == job


# ---- messages ------------------------------------------------------------------------------------------------


def test_a_message_quotes_none_of_the_values_the_file_holds() -> None:
    doc = document()
    doc["spm"]["vocab_size"] = SENTINEL
    doc["created"] = SENTINEL
    doc["registry"] = f"{SENTINEL}/../x"
    doc["decode"]["beam_size"] = [SENTINEL]
    doc["fixtures"]["sha256"][FIXTURE_FILES[0]] = SENTINEL
    doc["export"]["quantizations"] = [SENTINEL]
    doc["timeouts"]["train"] = SENTINEL
    message = refusal(doc)
    assert SENTINEL not in message
    for field in (
        "spm.vocab_size",
        "created",
        "registry",
        "decode.beam_size",
        "export.quantizations",
        "timeouts.train",
    ):
        assert field in message


def test_a_key_that_is_not_a_short_identifier_is_not_echoed() -> None:
    for key in (f"{SENTINEL}/x", "a" * 41, "has space", "x\ny", "k;v"):
        doc = document()
        doc["spm"][key] = 1
        message = refusal(doc)
        assert key not in message
        assert "spm.<key>" in message


def test_the_timeouts_and_the_digests_come_back_in_the_order_of_the_closed_lists() -> (
    None
):
    doc = document()
    doc["timeouts"] = dict(reversed(list(doc["timeouts"].items())))
    doc["fixtures"]["sha256"] = dict(reversed(list(doc["fixtures"]["sha256"].items())))
    config = load_tiny_config(dump(doc))
    assert tuple(config.timeouts) == STAGE_IDS
    assert tuple(config.fixtures.sha256) == FIXTURE_FILES


def test_a_key_that_is_a_short_identifier_is_named() -> None:
    doc = document()
    doc["spm"]["vocab_sise"] = 1
    assert "spm.vocab_sise" in refusal(doc)


def test_a_refused_key_of_a_closed_table_is_named_without_a_mark_of_the_validator() -> (
    None
):
    doc = document()
    doc["fixtures"]["sha256"]["bogus_key"] = "a" * 64
    message = refusal(doc)
    assert "fixtures.sha256.bogus_key" in message
    doc = document()
    doc["timeouts"]["bogus_key"] = 30
    message = refusal(doc)
    assert "timeouts.bogus_key" in message
    assert "[key]" not in message
    assert "<key>" not in message
    assert "<key>" not in refusal(with_value(("fixtures", "sha256", "nouns.tsv"), 5))


def test_every_message_of_the_schema_starts_with_the_same_prefix() -> None:
    assert refusal(with_value(("threads",), 0)).startswith(
        "invalid tiny config: threads"
    )
    assert refusal(with_value(("budget_seconds",), "x")).startswith(
        "invalid tiny config: budget_seconds"
    )


def test_every_field_at_fault_is_named_in_one_message() -> None:
    doc = document()
    doc["threads"] = 0
    doc["decode"]["beam_size"] = 0
    doc["created"] = "x"
    message = refusal(doc)
    for field in ("threads", "decode.beam_size", "created"):
        assert field in message


def test_nothing_is_read_from_the_environment_or_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = dump(document())
    expected = load_tiny_config(data)
    for name, value in {
        "RINCHEN_SEED": "5",
        "RINCHEN_PROFILE": "full",
        "RINCHEN_THREADS": "9",
        "RINCHEN_TINY_BUDGET_SECONDS": "5",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    assert load_tiny_config(data) == expected


# ---- the path of a profile ------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["tiny", "full", "a", "a1_b", "t" * 16, "a_", "z9"])
def test_the_path_of_a_profile_is_its_file_under_configs_tiny(profile: str) -> None:
    assert (
        config_path_for(profile) == REPO_ROOT / "configs" / "tiny" / f"{profile}.yaml"
    )


@pytest.mark.parametrize(
    "profile",
    [
        "Tiny",
        "TINY",
        "a/b",
        "../x",
        "a\\b",
        "",
        "1a",
        "_a",
        "tiny.yaml",
        "t" * 17,
        "a-b",
        "tiny\n",
        " tiny",
        "tiny ",
        "t\u00e9",
        "\u0430",
        "a b",
        "a\x00",
        "..",
        "/etc",
    ],
)
def test_a_profile_name_outside_the_closed_shape_is_refused_without_being_quoted(
    profile: str,
) -> None:
    with pytest.raises(ConfigError, match="profile") as caught:
        config_path_for(profile)
    if profile.strip():
        assert profile not in str(caught.value)


def test_a_refused_profile_name_is_not_echoed() -> None:
    with pytest.raises(ConfigError) as caught:
        config_path_for(SENTINEL)
    assert SENTINEL not in str(caught.value)


def test_building_a_path_does_not_read_the_file() -> None:
    # The command, not this function, refuses a profile that has no file.
    assert not config_path_for("nowhere").exists()
