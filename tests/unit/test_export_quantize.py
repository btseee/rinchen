"""Tests for `rinchen.export.quantize`: the quantisations an export offers and what they
load as. Quantisation is applied at conversion and recorded in `rinchen.json`; at load the
engine asks for what the file stores (`compute_type="default"`), never `auto`, which
quantises even a float32 export to int8 on a CPU (measured in the D11 spike)."""

from __future__ import annotations

import pytest

from rinchen.errors import ConfigError
from rinchen.export.quantize import (
    QUANTIZATIONS,
    effective_compute_type,
    require_compute_type,
    require_quantization,
)


def test_an_export_offers_four_quantizations_the_reference_first() -> None:
    assert QUANTIZATIONS == ("float32", "float16", "int8", "int8_float16")


@pytest.mark.parametrize("value", QUANTIZATIONS)
def test_an_offered_quantization_is_returned_as_it_is(value: str) -> None:
    assert require_quantization(value) == value


@pytest.mark.parametrize(
    "value", ["int16", "int4", "bfloat16", "int8_float32", "INT8", "auto", "", " int8"]
)
def test_a_quantization_that_is_not_offered_is_a_config_error_listing_the_four(
    value: str,
) -> None:
    with pytest.raises(ConfigError) as raised:
        require_quantization(value)

    assert "float32, float16, int8, int8_float16" in str(raised.value)


@pytest.mark.parametrize(
    ("quantization", "device", "loads_as"),
    [
        ("float32", "cpu", "float32"),
        ("float16", "cpu", "float32"),
        ("int8", "cpu", "int8_float32"),
        ("int8_float16", "cpu", "int8_float32"),
        ("float32", "cuda", "float32"),
        ("float16", "cuda", "float16"),
        ("int8", "cuda", "int8"),
        ("int8_float16", "cuda", "int8_float16"),
    ],
)
def test_the_compute_type_an_export_loads_as_on_each_device(
    quantization: str, device: str, loads_as: str
) -> None:
    assert effective_compute_type(quantization, device) == loads_as


def test_the_cuda_answer_is_the_stored_type_and_says_it_was_never_measured() -> None:
    assert effective_compute_type.__doc__ is not None
    assert "UNMEASURED" in effective_compute_type.__doc__


@pytest.mark.parametrize("device", ["auto", "gpu", "", "CPU", "cuda:0"])
def test_a_device_other_than_cpu_or_cuda_is_a_config_error(device: str) -> None:
    with pytest.raises(ConfigError, match="device"):
        effective_compute_type("float32", device)


def test_a_quantization_that_is_not_offered_has_no_effective_compute_type() -> None:
    with pytest.raises(ConfigError, match="float32, float16, int8, int8_float16"):
        effective_compute_type("int4", "cpu")


@pytest.mark.parametrize("value", ["auto", "AUTO", " auto ", "Auto"])
def test_the_automatic_compute_type_is_refused_because_it_quantizes_a_float32_export(
    value: str,
) -> None:
    with pytest.raises(ConfigError) as raised:
        require_compute_type(value)

    assert "auto" in str(raised.value)
    assert "int8" in str(raised.value)


@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_a_blank_compute_type_is_refused(value: str) -> None:
    with pytest.raises(ConfigError, match="blank"):
        require_compute_type(value)


@pytest.mark.parametrize(
    "value", ["default", "int8", "float32", "int8_float16", "float16", "int16"]
)
def test_a_named_compute_type_is_returned_as_it_is(value: str) -> None:
    assert require_compute_type(value) == value
