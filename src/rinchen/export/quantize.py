"""The quantisations an export offers, and what each loads as (D11).

Quantisation is applied AT CONVERSION and recorded in `rinchen.json`: the converter stores the
weights as the quantisation says. At load the engine asks CTranslate2 for what the file
stores (`compute_type="default"`), and the device falls back with CTranslate2's own warning
when it cannot run the stored type. `auto` is refused everywhere: on a CPU it quantises
EVERY export, a float32 one included, to `int8_float32` (measured in the D11 spike), so a
check or a gate that used it would measure int8 and call it float32.

`float32` is the parity reference and the default, `int8` the CPU product, `float16` and
`int8_float16` the GPU products: they are converted here and never run on a GPU (UNMEASURED).
The converter accepts four more types (`int8_float32`, `int8_bfloat16`, `int16`,
`bfloat16`); they are not offered, and one tuple entry each adds them if a need appears.
"""

from __future__ import annotations

from typing import Final

from rinchen.errors import ConfigError

QUANTIZATIONS: Final = ("float32", "float16", "int8", "int8_float16")

# What `compute_type="default"` loads each export as on a CPU (measured, CTranslate2 4.8.4):
# the types a CPU cannot run efficiently fall back with a warning on stderr.
_ON_CPU: Final = {
    "float32": "float32",
    "float16": "float32",
    "int8": "int8_float32",
    "int8_float16": "int8_float32",
}
_DEVICES: Final = ("cpu", "cuda")
_AUTOMATIC: Final = "auto"


def require_quantization(value: str) -> str:
    """`value` when it is one of `QUANTIZATIONS`.

    Raises:
        ConfigError: it is anything else; the message lists the four.
    """
    if value not in QUANTIZATIONS:
        raise ConfigError(
            f"quantization {value!r} is not offered: it must be one of "
            f"{', '.join(QUANTIZATIONS)}"
        )
    return value


def effective_compute_type(quantization: str, device: str) -> str:
    """The compute type an export of `quantization` loads as on `device`, `cpu` or `cuda`.

    On a CPU: `float32` and `float16` load as `float32`, `int8` and `int8_float16` as
    `int8_float32` (measured). On `cuda` it is the stored type itself (`float32`,
    `float16`, `int8`, `int8_float16`): that is CTranslate2's documented behaviour and it
    is UNMEASURED here, because no GPU was used.

    Raises:
        ConfigError: `quantization` is not offered, or `device` is not `cpu` or `cuda`
            (`auto` included: the device is named, never guessed).
    """
    require_quantization(quantization)
    if device not in _DEVICES:
        raise ConfigError(
            f"device {device!r} is not one of {', '.join(_DEVICES)}: the device is "
            "named, never chosen automatically"
        )
    return _ON_CPU[quantization] if device == "cpu" else quantization


def require_compute_type(value: str) -> str:
    """`value`, the compute type to load a model with, unless it is blank or `auto`.

    `default` loads what the file stores. Any other name is CTranslate2's to accept or
    refuse (an explicit type the device cannot run is a `ValueError` there).

    Raises:
        ConfigError: `value` is blank, or `auto`, which quantises a float32 export to
            int8 on a CPU.
    """
    if not value.strip():
        raise ConfigError("the compute type is blank: name one, or `default`")
    if value.strip().lower() == _AUTOMATIC:
        raise ConfigError(
            "the compute type `auto` is refused: it quantises even a float32 export to "
            "int8 on a CPU, so a result would not be the export's own; use `default` "
            "for what the file stores, or name a type"
        )
    return value
