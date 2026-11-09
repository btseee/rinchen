"""Tests for the tag placement of `rinchen.train.distill_export.export_checkpoint` (D10).

The placement of the direction tag (`src`: in the source, `tgt`: a forced prefix of the target)
is what a translator must know to use a model, and CTranslate2's files do not hold it. A
checkpoint our launcher wrote does: the `data` section of its saved `config.json` carries the
prefixes of every corpus. An export that states the other placement would make the converter
succeed and every translation come out in the wrong script, so the statement is checked
against the saved config (which was hashed before it was read), and a disagreement is refused
before anything is made. A saved config without a `data` section, or without a direction tag
in it, leaves the caller's statement as the only source.

The runs are fabricated by `tests.support.step_dirs` (`launcher_data`) and the converter is
`tests.support.distill_exports.fake_converter`: nothing starts a process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import ConfigError, DataError
from rinchen.train.distill_export import export_checkpoint
from tests.support.distill_exports import FakeConverter, fake_converter
from tests.support.step_dirs import (
    eole_model_section,
    launcher_data,
    make_averaged_dir,
    make_finished_run,
    tree_digest,
    write_spm_model,
)

# The `data` section Eole saved for the averaged teacher of the design spike (placement
# `src`, the corpus paths cut): the same keys in the same order, as the launcher wrote them.
SPIKE_SRC_DATA: dict[str, Any] = {
    "eng_Latn-khk_Cyrl.auth": {
        "path_src": "/work/data/auth/eng_Latn-khk_Cyrl.auth.src",
        "tgt_prefix": "",
        "path_tgt": "/work/data/auth/eng_Latn-khk_Cyrl.auth.tgt",
        "transforms": ["sentencepiece", "prefix", "filtertoolong"],
        "src_prefix": "<2khk_Cyrl>",
        "path_align": None,
        "weight": 22,
    },
    "eng_Latn-khk_Cyrl.bt": {
        "path_src": "/work/data/bt/eng_Latn-khk_Cyrl.bt.src",
        "tgt_prefix": "",
        "path_tgt": "/work/data/bt/eng_Latn-khk_Cyrl.bt.tgt",
        "transforms": ["sentencepiece", "prefix", "filtertoolong"],
        "src_prefix": "<2khk_Cyrl> <bt>",
        "path_align": None,
        "weight": 6,
    },
    "valid": {
        "path_src": "/work/valid.src",
        "path_tgt": "/work/valid.tgt",
    },
}
MODEL = eole_model_section(encoder_layers=2, decoder_layers=1)


@pytest.fixture
def spm_model(tmp_path: Path) -> Path:
    return write_spm_model(tmp_path / "spm")


@pytest.fixture
def converter(monkeypatch: pytest.MonkeyPatch) -> FakeConverter:
    return fake_converter(monkeypatch)


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "export"


def run_with(tmp_path: Path, data: dict[str, Any] | None) -> Path:
    """The final step of a finished run whose saved config holds `data`."""
    return make_finished_run(tmp_path / "teacher", model=MODEL, data=data).final


def export(checkpoint: Path, target: Path, spm_model: Path, placement: Any) -> Any:
    return export_checkpoint(
        checkpoint, target, spm_model=spm_model, tag_placement=placement
    )


def assert_refused_with_nothing_made(
    checkpoint: Path,
    target: Path,
    spm_model: Path,
    placement: str,
    converter: FakeConverter,
    error: type[Exception],
) -> str:
    before = tree_digest(checkpoint.parent.parent)
    with pytest.raises(error) as raised:
        export(checkpoint, target, spm_model, placement)
    assert not target.exists()
    assert converter.calls == []
    assert tree_digest(checkpoint.parent.parent) == before
    return str(raised.value)


# ---- a statement that disagrees with the checkpoint is refused ---------------------------


def test_a_source_tag_checkpoint_stated_as_target_tag_is_refused_naming_both_values(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    checkpoint = run_with(tmp_path, SPIKE_SRC_DATA)

    message = assert_refused_with_nothing_made(
        checkpoint, out_dir, spm_model, "tgt", converter, ConfigError
    )

    assert "'tgt'" in message
    assert "'src'" in message
    assert "data.eng_Latn-khk_Cyrl.auth.src_prefix" in message
    assert "'<2khk_Cyrl>'" in message
    assert str(checkpoint / "config.json") in message


def test_a_target_tag_checkpoint_stated_as_source_tag_is_refused_naming_both_values(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    checkpoint = run_with(tmp_path, launcher_data("tgt"))

    message = assert_refused_with_nothing_made(
        checkpoint, out_dir, spm_model, "src", converter, ConfigError
    )

    assert "'src'" in message
    assert "'tgt'" in message
    assert "data.eng_Latn-khk_Cyrl.auth.tgt_prefix" in message
    assert "'<2khk_Cyrl>'" in message


def test_an_averaged_model_is_checked_like_a_step(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    averaged = make_averaged_dir(
        tmp_path / "averaged", model=MODEL, data=launcher_data("src")
    )

    message = assert_refused_with_nothing_made(
        averaged, out_dir, spm_model, "tgt", converter, ConfigError
    )

    assert "'tgt'" in message
    assert "'src'" in message


# ---- a statement that agrees is recorded as it is -----------------------------------------


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_the_placement_the_checkpoint_records_is_exported(
    placement: str,
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
) -> None:
    checkpoint = run_with(tmp_path, launcher_data(placement))

    record = export(checkpoint, out_dir, spm_model, placement)

    assert record.tag_placement == placement
    assert len(converter.calls) == 1


def test_an_averaged_model_with_the_recorded_placement_is_exported(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    averaged = make_averaged_dir(
        tmp_path / "averaged", model=MODEL, data=launcher_data("tgt")
    )

    assert export(averaged, out_dir, spm_model, "tgt").tag_placement == "tgt"


def test_the_spikes_saved_data_section_is_read_as_a_source_tag_checkpoint(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    checkpoint = run_with(tmp_path, SPIKE_SRC_DATA)

    assert export(checkpoint, out_dir, spm_model, "src").tag_placement == "src"


@pytest.mark.parametrize(
    ("placement", "marker"),
    [
        ("src", "<ft>"),
        ("tgt", "<ft>"),
        ("src", "<doc>"),
        ("src", "<bt>"),
        ("tgt", "<bt>"),
    ],
)
def test_an_origin_marker_beside_the_tag_does_not_hide_the_placement(
    placement: str,
    marker: str,
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
) -> None:
    # A student's KD corpus carries `<ft>` (as a back-translated one carries `<bt>`) and a
    # document-level run `<doc>`: on the source side with the tag, or, for placement `tgt`,
    # alone as the source prefix.
    data = launcher_data("tgt", (("eng_Latn-khk_Cyrl.kd", "<2khk_Cyrl>", marker),))
    if placement == "src":
        data = launcher_data("src", (("eng_Latn-khk_Cyrl.kd", "<2khk_Cyrl>", marker),))
    checkpoint = run_with(tmp_path, data)
    other = "tgt" if placement == "src" else "src"

    assert export(checkpoint, out_dir, spm_model, placement).tag_placement == placement
    with pytest.raises(
        ConfigError, match=f"'{other}'.*'{placement}'|'{placement}'.*'{other}'"
    ):
        export(checkpoint, out_dir.with_name("export2"), spm_model, other)


def test_a_single_corpus_is_enough_to_tell_the_placement(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    only_one = launcher_data("tgt", (("eng_Latn-khk_Cyrl.auth", "<2khk_Cyrl>", None),))
    checkpoint = run_with(tmp_path, only_one)

    with pytest.raises(ConfigError, match="'src'"):
        export(checkpoint, out_dir, spm_model, "src")


# ---- no statement can be checked: the caller's stands ---------------------------------------


@pytest.mark.parametrize("placement", ["src", "tgt"])
@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"valid": {"path_src": "/v.src", "path_tgt": "/v.tgt"}},
        {
            "corpus": {
                "path_src": "/c.src",
                "path_tgt": "/c.tgt",
                "src_prefix": "",
                "tgt_prefix": "",
            }
        },
        {"corpus": {"path_src": "/c.src", "path_tgt": "/c.tgt", "src_prefix": "<bt>"}},
    ],
    ids=["no data", "empty data", "valid only", "no tag", "marker only"],
)
def test_a_saved_config_without_a_direction_tag_leaves_the_callers_statement(
    placement: str,
    data: dict[str, Any] | None,
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
) -> None:
    checkpoint = run_with(tmp_path, data)

    assert export(checkpoint, out_dir, spm_model, placement).tag_placement == placement


# ---- a saved config that contradicts itself or is not what Eole writes -----------------------


def test_a_corpus_with_a_direction_tag_on_both_sides_is_refused(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    data = launcher_data("src")
    data["eng_Latn-khk_Cyrl.auth"]["tgt_prefix"] = "<2khk_Cyrl>"
    checkpoint = run_with(tmp_path, data)

    message = assert_refused_with_nothing_made(
        checkpoint, out_dir, spm_model, "src", converter, DataError
    )

    assert "eng_Latn-khk_Cyrl.auth" in message
    assert "both" in message


def test_corpora_that_place_the_tag_differently_are_refused(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    data = launcher_data("src")
    data["eng_Latn-khk_Cyrl.bt"] = launcher_data("tgt")["eng_Latn-khk_Cyrl.bt"]
    checkpoint = run_with(tmp_path, data)

    message = assert_refused_with_nothing_made(
        checkpoint, out_dir, spm_model, "src", converter, DataError
    )

    assert "eng_Latn-khk_Cyrl.auth" in message
    assert "eng_Latn-khk_Cyrl.bt" in message


@pytest.mark.parametrize(
    "data",
    [
        [],
        "src",
        {"corpus": "<2khk_Cyrl>"},
        {"corpus": {"src_prefix": 7, "tgt_prefix": ""}},
        {"corpus": {"src_prefix": "", "tgt_prefix": ["<2khk_Cyrl>"]}},
    ],
    ids=["list", "text", "entry is text", "prefix is a number", "prefix is a list"],
)
def test_a_saved_data_section_that_is_not_what_eole_writes_is_refused(
    data: Any,
    tmp_path: Path,
    spm_model: Path,
    converter: FakeConverter,
    out_dir: Path,
) -> None:
    checkpoint = run_with(tmp_path, data)

    message = assert_refused_with_nothing_made(
        checkpoint, out_dir, spm_model, "src", converter, DataError
    )

    assert str(checkpoint / "config.json") in message


def test_the_refusal_comes_before_the_converter_and_before_the_output_directory_is_made(
    tmp_path: Path, spm_model: Path, converter: FakeConverter, out_dir: Path
) -> None:
    checkpoint = run_with(tmp_path, launcher_data("src"))

    with pytest.raises(ConfigError):
        export(checkpoint, out_dir, spm_model, "tgt")

    assert converter.calls == []
    assert not out_dir.exists()
    assert not Path(f"{out_dir}.export.log").exists()
