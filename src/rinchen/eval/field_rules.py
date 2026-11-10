"""The checks of one field of a document that D13 reads back (`result.json`, `run.json` and the
robustness files of D12): a value of the right type and shape, or a `ValueError` that names the
field and NEVER quotes the value (a file may hold a text, a path or a secret). The readers wrap
the `ValueError` into a `DataError` that says which kind of file it was.

A field is named by its dotted path in the document (`bootstrap.units`, `directions.2.state`);
a key of the file is named only when it is a key of the format.

`load_document` is the part every reader begins with: UTF-8, JSON, an object at the top and no
key repeated at any depth (D12's `parse_object` keeps a repeated key so that this can say so).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final, Literal, overload

from rinchen.config.languages import known_language_codes
from rinchen.errors import DataError
from rinchen.eval.robust_suite_rules import JsonObject, is_id, parse_object
from rinchen.eval.suite_rules import FORBIDDEN_LANGUAGES
from rinchen.eval.systems import Scalar, SystemRecord

__all__ = [
    "SYSTEM_FIELDS",
    "document_bytes",
    "load_document",
    "problem",
    "require_choice",
    "require_digest",
    "require_direction",
    "require_fields",
    "require_flag",
    "require_id",
    "require_integer",
    "require_number",
    "require_object",
    "require_text",
    "system_from",
]

SYSTEM_FIELDS: Final = (
    "kind",
    "name",
    "model_sha256",
    "manifest_sha256",
    "options",
    "outputs_sha256",
)
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_KINDS: Final = ("rinchen", "baseline", "hypotheses")


def problem(field: str | None, reason: str) -> ValueError:
    """The error for a field that breaks a rule: `field` is a dotted path of the format,
    never a value of the file."""
    return ValueError(reason if field is None else f"field {field!r}: {reason}")


def document_bytes(document: Mapping[str, Any]) -> bytes:
    """The UTF-8 bytes of a document: sorted keys, two-space indent, text as it is, one final
    LF. Floats are written as Python writes them and rounded nowhere.

    Raises:
        ValueError: a value is not a finite number (JSON has no NaN).
    """
    text = json.dumps(
        document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    )
    return f"{text}\n".encode()


def load_document(data: bytes, label: str) -> JsonObject:
    """The JSON object in `data`, `label` being what the file should have been (`an evaluation
    result`).

    Raises:
        DataError: not UTF-8, not JSON, not an object, or a key repeated at any depth.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise DataError(f"not {label}: the file is not UTF-8") from None
    try:
        parsed = json.loads(text, object_pairs_hook=parse_object)
    except ValueError, RecursionError:
        raise DataError(f"not {label}: the file is not valid JSON") from None
    if not isinstance(parsed, JsonObject):
        raise DataError(f"not {label}: the file is not a JSON object")
    if _repeats_a_key(parsed):
        raise DataError(f"not {label}: a field appears more than once")
    return parsed


def _repeats_a_key(root: object) -> bool:
    pending = [root]
    while pending:
        node = pending.pop()
        if isinstance(node, JsonObject) and node.repeated:
            return True
        if isinstance(node, dict):
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return False


def require_object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise problem(field, "must be an object")
    return value


def require_fields(
    value: object, field: str | None, names: Sequence[str]
) -> Mapping[str, Any]:
    """`value` as an object with exactly the fields `names`: a missing one is named, an extra
    one is not (it is a key of the file)."""
    if not isinstance(value, Mapping):
        raise problem(field, "must be an object")
    for name in names:
        if name not in value:
            raise problem(field, f"the field {name!r} is missing")
    if len(value) != len(names):
        raise problem(field, "has a field that is not part of the format")
    return value


def require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise problem(field, "must be a non-empty text")
    return value


def require_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not is_id(value):
        raise problem(
            field, "must be 1 to 64 characters: letters, digits, '.', '_' and '-'"
        )
    return value


@overload
def require_digest(
    value: object, field: str, *, nullable: Literal[False] = ...
) -> str: ...
@overload
def require_digest(value: object, field: str, *, nullable: bool) -> str | None: ...
def require_digest(value: object, field: str, *, nullable: bool = False) -> str | None:
    """A SHA-256 as 64 lowercase hexadecimal characters (or null when `nullable`)."""
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise problem(
            field,
            "must be 64 lowercase hexadecimal characters"
            + (" or null" if nullable else ""),
        )
    return value


def require_integer(
    value: object, field: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    """An integer (a boolean is not one) of at least `minimum` and at most `maximum`."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = (
            f"at least {minimum}" if maximum is None else f"from {minimum} to {maximum}"
        )
        raise problem(field, f"must be an integer {bound}")
    return value


@overload
def require_number(
    value: object, field: str, *, nullable: Literal[False] = ...
) -> float: ...
@overload
def require_number(value: object, field: str, *, nullable: bool) -> float | None: ...
def require_number(
    value: object, field: str, *, nullable: bool = False
) -> float | None:
    """A finite number (a boolean is not one), as a float; null when `nullable`."""
    if value is None and nullable:
        return None
    return _finite(value, field, " or null" if nullable else "")


def _finite(value: object, field: str, alternative: str = "") -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise problem(field, f"must be a finite number{alternative}")
    return float(value)


def require_direction(value: object, field: str) -> str:
    """`<src>-<tgt>`: two different language codes of the registry. `khk_Mong` is refused by
    name, before the registry is asked; any other string is never quoted."""
    rule = "must be two language codes of the registry joined by '-'"
    if not isinstance(value, str):
        raise problem(field, rule)
    src, separator, tgt = value.partition("-")
    if not separator or "-" in tgt:
        raise problem(field, rule)
    for code in (src, tgt):
        if code in FORBIDDEN_LANGUAGES:
            raise problem(
                field, f"{code} is refused by name: Inner Mongolian data is never used"
            )
        if code not in known_language_codes():
            raise problem(field, rule)
    if src == tgt:
        raise problem(field, "must name two different languages")
    return value


def require_flag(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise problem(field, "must be true or false")
    return value


def require_choice(value: object, field: str, choices: Collection[str]) -> str:
    """One of `choices`, which are constants of the format and may be named."""
    if not isinstance(value, str) or value not in choices:
        raise problem(field, f"must be one of {', '.join(choices)}")
    return value


def system_from(raw: object, field: str) -> SystemRecord:
    """The system record of a document: its six fields, each of its type."""
    system = require_fields(raw, field, SYSTEM_FIELDS)
    kind = require_choice(system["kind"], f"{field}.kind", _KINDS)
    options = require_object(system["options"], f"{field}.options")
    checked: dict[str, Scalar] = {}
    for name, option in options.items():
        if isinstance(option, bool | str | int):
            checked[name] = option
        else:
            checked[name] = _finite(option, f"{field}.options")
    return SystemRecord(
        kind=kind,  # type: ignore[arg-type]
        name=require_id(system["name"], f"{field}.name"),
        model_sha256=require_digest(
            system["model_sha256"], f"{field}.model_sha256", nullable=True
        ),
        manifest_sha256=require_digest(
            system["manifest_sha256"], f"{field}.manifest_sha256", nullable=True
        ),
        options=checked,
        outputs_sha256=require_digest(
            system["outputs_sha256"], f"{field}.outputs_sha256", nullable=True
        ),
    )
