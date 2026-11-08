"""Tests for `rinchen.config.ablation`: the ablation matrix file (`name`, `base`, the
`axes` of temperature, back-translation ratio and tag placement, the seeds, and the
limits) and the loader that turns it into an `AblationMatrix` or a `ConfigError`.

The shipped matrices and the variants they expand to are tested in
`test_eole_configs_shipped.py` and `test_train_variants.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rinchen.config.ablation import (
    AblationMatrix,
    Axes,
    load_ablation,
    require_placement,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError

SHIPPED_TINY = REPO_ROOT / "configs" / "eole" / "ablation_tiny.yaml"
SHIPPED_TINY_BASE = REPO_ROOT / "configs" / "eole" / "multilingual_tiny.yaml"


def _write(tmp_path: Path, document: Any) -> Path:
    """Write `document` as YAML (or verbatim when it is already text)."""
    path = tmp_path / "ablation.yaml"
    text = document if isinstance(document, str) else yaml.safe_dump(document)
    path.write_text(text, encoding="utf-8")
    return path


def _document(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    """The smallest valid matrix: a name and a base file that exists."""
    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\n", encoding="utf-8")
    return {"name": "t", "base": str(base)} | changes


def _axes(**axes: Any) -> dict[str, Any]:
    return {"axes": axes}


def _message(tmp_path: Path, **changes: Any) -> str:
    """What `load_ablation` says about the minimal matrix with `changes` applied."""
    with pytest.raises(ConfigError) as raised:
        load_ablation(_write(tmp_path, _document(tmp_path, **changes)))
    return str(raised.value)


# --- the shipped tiny matrix and the defaults -------------------------------------


def test_the_shipped_tiny_matrix_has_the_values_of_the_spec() -> None:
    matrix = load_ablation(SHIPPED_TINY)

    assert matrix.name == "tiny"
    assert matrix.base == SHIPPED_TINY_BASE
    assert matrix.axes.temperature == [0.5]
    assert matrix.axes.bt_ratio == [0.5]
    assert matrix.axes.tag_placement == ["src", "tgt"]
    assert matrix.seeds == [1234]
    assert matrix.weight_total == 100
    assert matrix.dev_max_lines == 1000
    assert matrix.max_variants == 64
    assert matrix.overrides == {}


def test_a_matrix_with_a_name_and_a_base_takes_the_documented_defaults(
    tmp_path: Path,
) -> None:
    matrix = load_ablation(_write(tmp_path, _document(tmp_path)))

    assert matrix.name == "t"
    assert matrix.base == tmp_path / "base.yaml"
    assert matrix.axes == Axes(temperature=[1.0], bt_ratio=[0.0], tag_placement=["src"])
    assert matrix.seeds == [1234]
    assert matrix.weight_total == 100
    assert matrix.dev_max_lines == 1000
    assert matrix.max_variants == 64
    assert matrix.overrides == {}


def test_every_value_given_replaces_its_default(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        axes={"temperature": [0.3, 1], "bt_ratio": [0, 0.25], "tag_placement": ["tgt"]},
        seeds=[0, 7],
        weight_total=60,
        dev_max_lines=5,
        max_variants=8,
        overrides={"training": {"train_steps": 10}},
    )

    matrix = load_ablation(_write(tmp_path, document))

    assert matrix.axes.temperature == [0.3, 1.0]
    assert matrix.axes.bt_ratio == [0.0, 0.25]
    assert matrix.axes.tag_placement == ["tgt"]
    assert matrix.seeds == [0, 7]
    assert matrix.weight_total == 60
    assert matrix.dev_max_lines == 5
    assert matrix.max_variants == 8
    assert matrix.overrides == {"training": {"train_steps": 10}}


def test_an_axes_section_leaves_the_axes_it_does_not_name_at_their_defaults(
    tmp_path: Path,
) -> None:
    matrix = load_ablation(
        _write(tmp_path, _document(tmp_path, **_axes(bt_ratio=[0.5])))
    )

    assert matrix.axes.temperature == [1.0]
    assert matrix.axes.bt_ratio == [0.5]
    assert matrix.axes.tag_placement == ["src"]


def test_each_matrix_gets_its_own_default_lists(tmp_path: Path) -> None:
    first = load_ablation(_write(tmp_path, _document(tmp_path)))
    first.seeds.append(99)
    first.axes.temperature.append(9.0)

    second = load_ablation(_write(tmp_path, _document(tmp_path)))

    assert second.seeds == [1234]
    assert second.axes.temperature == [1.0]


def test_a_matrix_is_frozen(tmp_path: Path) -> None:
    matrix = load_ablation(_write(tmp_path, _document(tmp_path)))

    with pytest.raises(ValidationError):
        matrix.name = "other"  # type: ignore[misc]


def test_the_matrix_model_can_be_built_directly(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\n", encoding="utf-8")

    matrix = AblationMatrix(name="direct", base=base)

    assert matrix.axes == Axes()
    assert matrix.base == base


# --- the base file -----------------------------------------------------------------


def test_a_relative_base_means_the_same_file_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    path = _write(
        tmp_path, {"name": "t", "base": "configs/eole/multilingual_tiny.yaml"}
    )

    assert load_ablation(path).base == SHIPPED_TINY_BASE


def test_an_absolute_base_is_kept_as_written(tmp_path: Path) -> None:
    matrix = load_ablation(_write(tmp_path, _document(tmp_path)))

    assert matrix.base == tmp_path / "base.yaml"


def test_a_missing_base_file_is_an_error_when_the_matrix_is_loaded(
    tmp_path: Path,
) -> None:
    message = _message(tmp_path, base=str(tmp_path / "no-such-base.yaml"))

    assert "base" in message
    assert "no-such-base.yaml" in message
    assert "not found" in message


def test_a_relative_base_that_does_not_exist_is_reported_at_its_repo_path(
    tmp_path: Path,
) -> None:
    message = _message(tmp_path, base="configs/eole/no_such_base.yaml")

    assert str(REPO_ROOT / "configs" / "eole" / "no_such_base.yaml") in message


def test_a_base_that_is_a_directory_is_an_error(tmp_path: Path) -> None:
    message = _message(tmp_path, base=str(tmp_path))

    assert "not found" in message


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_base_is_an_error(tmp_path: Path, blank: str) -> None:
    assert "base must not be empty or blank" in _message(tmp_path, base=blank)


def test_a_matrix_without_a_base_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, {"name": "t"})

    with pytest.raises(ConfigError, match=r"base\n\s+Field required"):
        load_ablation(path)


def test_a_matrix_without_a_name_is_an_error(tmp_path: Path) -> None:
    document = _document(tmp_path)
    del document["name"]

    with pytest.raises(ConfigError, match=r"name\n\s+Field required"):
        load_ablation(_write(tmp_path, document))


@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_name_is_an_error(tmp_path: Path, blank: str) -> None:
    assert "must not be blank" in _message(tmp_path, name=blank)


# --- the axes ----------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0, 0.0, -0.5])
def test_a_temperature_that_is_not_positive_is_an_error(
    tmp_path: Path, temperature: float
) -> None:
    message = _message(tmp_path, **_axes(temperature=[temperature]))

    assert "axes.temperature.0" in message
    assert "greater than 0" in message


@pytest.mark.parametrize("value", [".inf", ".nan"])
def test_a_temperature_that_is_not_finite_is_an_error(
    tmp_path: Path, value: str
) -> None:
    (tmp_path / "base.yaml").write_text("seed: 1\n", encoding="utf-8")
    path = _write(
        tmp_path,
        f"name: t\nbase: {tmp_path / 'base.yaml'}\naxes:\n  temperature: [{value}]\n",
    )

    with pytest.raises(ConfigError, match="axes.temperature.0"):
        load_ablation(path)


@pytest.mark.parametrize("ratio", [-0.1, -1])
def test_a_negative_bt_ratio_is_an_error(tmp_path: Path, ratio: float) -> None:
    message = _message(tmp_path, **_axes(bt_ratio=[ratio]))

    assert "axes.bt_ratio.0" in message
    assert "greater than or equal to 0" in message


def test_a_bt_ratio_of_zero_is_allowed(tmp_path: Path) -> None:
    document = _document(tmp_path, **_axes(bt_ratio=[0]))

    assert load_ablation(_write(tmp_path, document)).axes.bt_ratio == [0.0]


def test_a_bt_ratio_of_negative_zero_is_an_error_that_says_to_write_zero(
    tmp_path: Path,
) -> None:
    # It equals 0.0 but prints as `-0` in a variant id, so `0.0` and `-0.0` would
    # name two variants that train alike.
    message = _message(tmp_path, **_axes(bt_ratio=[-0.0]))

    assert "axes.bt_ratio.0" in message
    assert "negative zero" in message
    assert "0.0" in message


def test_a_bt_ratio_of_negative_zero_next_to_zero_is_still_an_error(
    tmp_path: Path,
) -> None:
    assert "negative zero" in _message(tmp_path, **_axes(bt_ratio=[0.0, -0.0]))


def test_the_axes_model_refuses_a_negative_zero_ratio_too() -> None:
    with pytest.raises(ValidationError, match="negative zero"):
        Axes(bt_ratio=[-0.0])


def test_a_temperature_of_negative_zero_is_not_positive(tmp_path: Path) -> None:
    assert "greater than 0" in _message(tmp_path, **_axes(temperature=[-0.0]))


@pytest.mark.parametrize("axis", ["temperature", "bt_ratio"])
def test_an_exponent_yaml_reads_as_text_is_reported_with_the_spelling_that_works(
    tmp_path: Path, axis: str
) -> None:
    (tmp_path / "base.yaml").write_text("seed: 1\n", encoding="utf-8")
    path = _write(
        tmp_path,
        f"name: t\nbase: {tmp_path / 'base.yaml'}\naxes:\n  {axis}: [1e-5]\n",
    )

    with pytest.raises(ConfigError) as raised:
        load_ablation(path)

    message = str(raised.value)
    assert f"axes.{axis}.0" in message
    assert "valid number" in message
    assert "'1e-5'" in message
    assert "1.0e-5" in message


@pytest.mark.parametrize("axis", ["temperature", "bt_ratio"])
def test_an_exponent_with_a_point_and_a_sign_is_a_number_in_yaml(
    tmp_path: Path, axis: str
) -> None:
    (tmp_path / "base.yaml").write_text("seed: 1\n", encoding="utf-8")
    path = _write(
        tmp_path,
        f"name: t\nbase: {tmp_path / 'base.yaml'}\naxes:\n  {axis}: [1.0e-5, 2.5e+3]\n",
    )

    assert getattr(load_ablation(path).axes, axis) == [1e-05, 2500.0]


@pytest.mark.parametrize("text", ["abc", "", "nan-ish"])
def test_a_text_that_is_not_a_number_keeps_the_plain_type_error(
    tmp_path: Path, text: str
) -> None:
    message = _message(tmp_path, **_axes(temperature=[text]))

    assert "valid number" in message
    assert "1.0e-5" not in message


@pytest.mark.parametrize("placement", ["both", "middle", "SRC", "", 1])
def test_a_placement_other_than_src_or_tgt_is_an_error(
    tmp_path: Path, placement: Any
) -> None:
    message = _message(tmp_path, **_axes(tag_placement=[placement]))

    assert "axes.tag_placement.0" in message
    assert "'src' or 'tgt'" in message


def test_a_suffix_placement_is_an_error_that_says_it_is_not_supported(
    tmp_path: Path,
) -> None:
    message = _message(tmp_path, **_axes(tag_placement=["src", "suffix"]))

    assert "axes.tag_placement.1" in message
    assert "suffix placement is not supported" in message
    assert "KeyError" in message


@pytest.mark.parametrize(
    ("axis", "values"),
    [
        ("temperature", [0.5, 1.0, 0.5]),
        ("bt_ratio", [0.0, 0.0]),
        ("tag_placement", ["src", "tgt", "src"]),
    ],
)
def test_a_value_repeated_in_an_axis_is_an_error(
    tmp_path: Path, axis: str, values: list[Any]
) -> None:
    message = _message(tmp_path, **_axes(**{axis: values}))

    assert f"axes.{axis}" in message
    assert "duplicate" in message


def test_a_seed_repeated_is_an_error(tmp_path: Path) -> None:
    message = _message(tmp_path, seeds=[1, 2, 1])

    assert "seeds" in message
    assert "duplicate" in message


def test_two_values_that_print_alike_in_an_id_are_duplicates(tmp_path: Path) -> None:
    # Variant ids print a number with six significant digits ("g"): these two would
    # name two variants alike.
    message = _message(tmp_path, **_axes(temperature=[0.1234567, 0.1234568]))

    assert "duplicate" in message
    assert "0.123457" in message


def test_an_integer_and_a_float_of_one_value_are_duplicates(tmp_path: Path) -> None:
    assert "duplicate" in _message(tmp_path, **_axes(temperature=[1, 1.0]))


@pytest.mark.parametrize("axis", ["temperature", "bt_ratio", "tag_placement"])
def test_an_empty_axis_is_an_error(tmp_path: Path, axis: str) -> None:
    message = _message(tmp_path, **_axes(**{axis: []}))

    assert f"axes.{axis}" in message
    assert "at least 1 item" in message


def test_an_empty_seed_list_is_an_error(tmp_path: Path) -> None:
    message = _message(tmp_path, seeds=[])

    assert "seeds" in message
    assert "at least 1 item" in message


def test_an_axis_that_is_not_a_list_is_an_error(tmp_path: Path) -> None:
    assert "axes.temperature" in _message(tmp_path, **_axes(temperature=0.5))


def test_an_unknown_field_in_axes_is_an_error(tmp_path: Path) -> None:
    message = _message(tmp_path, **_axes(temprature=[0.5]))

    assert "axes.temprature" in message
    assert "Extra inputs are not permitted" in message


def test_an_unknown_top_level_field_is_an_error(tmp_path: Path) -> None:
    message = _message(tmp_path, seed=[1234])

    assert "seed\n" in message
    assert "Extra inputs are not permitted" in message


# --- the seeds and the limits ------------------------------------------------------


@pytest.mark.parametrize("seed", [-1, -1234])
def test_a_negative_seed_is_an_error(tmp_path: Path, seed: int) -> None:
    message = _message(tmp_path, seeds=[seed])

    assert "seeds.0" in message
    assert "greater than or equal to 0" in message


@pytest.mark.parametrize("seed", [1.5, "7", True, None])
def test_a_seed_that_is_not_an_integer_is_an_error(tmp_path: Path, seed: Any) -> None:
    message = _message(tmp_path, seeds=[seed])

    assert "seeds.0" in message
    assert "valid integer" in message


def test_a_seed_of_zero_is_allowed(tmp_path: Path) -> None:
    assert load_ablation(_write(tmp_path, _document(tmp_path, seeds=[0]))).seeds == [0]


@pytest.mark.parametrize("field", ["weight_total", "dev_max_lines", "max_variants"])
@pytest.mark.parametrize("value", [0, -1])
def test_a_limit_below_one_is_an_error(tmp_path: Path, field: str, value: int) -> None:
    message = _message(tmp_path, **{field: value})

    assert f"{field}\n" in message
    assert "greater than 0" in message


@pytest.mark.parametrize("field", ["weight_total", "dev_max_lines", "max_variants"])
@pytest.mark.parametrize("value", [1.5, "10", True])
def test_a_limit_that_is_not_an_integer_is_an_error(
    tmp_path: Path, field: str, value: Any
) -> None:
    message = _message(tmp_path, **{field: value})

    assert f"{field}\n" in message
    assert "valid integer" in message


def test_an_expansion_above_max_variants_is_an_error_that_gives_the_count(
    tmp_path: Path,
) -> None:
    message = _message(
        tmp_path,
        axes={
            "temperature": [0.3, 0.5, 1.0],
            "bt_ratio": [0.0, 0.5],
            "tag_placement": ["src", "tgt"],
        },
        seeds=[1, 2],
        max_variants=23,
    )

    # 3 * 2 * 2 * 2
    assert "24 variants" in message
    assert "max_variants (23)" in message


def test_an_expansion_of_exactly_max_variants_is_allowed(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        axes={"temperature": [0.3, 0.5, 1.0], "tag_placement": ["src", "tgt"]},
        max_variants=6,
    )

    assert load_ablation(_write(tmp_path, document)).max_variants == 6


def test_overrides_must_be_a_mapping(tmp_path: Path) -> None:
    message = _message(tmp_path, overrides=["training"])

    assert "overrides" in message


# --- strict types ------------------------------------------------------------------


@pytest.mark.parametrize(
    "document_axes",
    [
        {"temperature": ["0.5"]},
        {"bt_ratio": ["0.5"]},
        {"temperature": [True]},
        {"bt_ratio": [False]},
        {"temperature": [None]},
    ],
)
def test_a_quoted_number_or_a_boolean_is_not_a_number(
    tmp_path: Path, document_axes: dict[str, Any]
) -> None:
    message = _message(tmp_path, axes=document_axes)

    assert "valid number" in message


def test_a_name_that_is_a_number_is_an_error(tmp_path: Path) -> None:
    assert "valid string" in _message(tmp_path, name=7)


# --- the file ----------------------------------------------------------------------


def test_a_missing_matrix_file_is_a_config_error_that_names_it(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found") as raised:
        load_ablation(tmp_path / "nonexistent.yaml")

    assert "nonexistent.yaml" in str(raised.value)


@pytest.mark.parametrize(
    ("text", "what"),
    [
        ("- a\n- b\n", "list"),
        ("just text\n", "str"),
        ("", "NoneType"),
        ("7\n", "int"),
    ],
)
def test_a_top_level_that_is_not_a_mapping_is_an_error_that_names_its_type(
    tmp_path: Path, text: str, what: str
) -> None:
    with pytest.raises(ConfigError, match="top level must be a mapping") as raised:
        load_ablation(_write(tmp_path, text))

    assert what in str(raised.value)


def test_a_syntax_error_is_a_config_error_with_the_yaml_error_as_its_cause(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="ablation.yaml") as raised:
        load_ablation(_write(tmp_path, "axes: [0.5\n"))

    assert isinstance(raised.value.__cause__, yaml.YAMLError)


def test_a_file_that_is_not_utf8_is_a_config_error_with_the_decode_error_as_its_cause(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ablation.yaml"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ConfigError, match="ablation.yaml") as raised:
        load_ablation(path)

    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


def test_a_field_violation_chains_the_validation_error(tmp_path: Path) -> None:
    path = _write(tmp_path, _document(tmp_path, seeds=[-1]))

    with pytest.raises(ConfigError, match="ablation.yaml") as raised:
        load_ablation(path)

    assert isinstance(raised.value.__cause__, ValidationError)


def test_the_loader_is_a_config_module_that_never_imports_the_train_package() -> None:
    # In a fresh interpreter, so that what other tests imported does not count.
    probe = (
        "import sys; import rinchen.config.ablation; "
        "bad = sorted(m for m in sys.modules if m.startswith('rinchen.train')); "
        "print(bad); sys.exit(1 if bad else 0)"
    )

    done = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("placement", ["src", "tgt"])
def test_a_supported_placement_passes_the_placement_check(placement: str) -> None:
    require_placement(placement)


@pytest.mark.parametrize("placement", ["suffix", "SRC", "", None, 1, ["src"]])
def test_any_other_placement_is_a_value_error_that_lists_the_supported_ones(
    placement: object,
) -> None:
    with pytest.raises(ValueError, match=r"placement must be one of src, tgt, got "):
        require_placement(placement)
