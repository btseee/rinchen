"""Walks a report document along its schema, and changes it at a place (D15 Task 6).

`closed_places` lists every mapping of a document that the schema holds to a closed set of keys, with
its path, so that a test can add an unknown key at each one in turn; `place_of` writes a path the way the
messages of `rinchen.tiny.report_shapes` do; `changed` returns a copy of the document with one value
replaced, one key added or one key removed at a path.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

from rinchen.tiny.report_shapes import Keys, Nullable, Seq, Table

FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures" / "tiny"
GOLDEN_JSON: Final = FIXTURES / "golden_report.json"
GOLDEN_MD: Final = FIXTURES / "golden_report.md"

Path_ = tuple[str | int, ...]


def golden_document() -> dict[str, Any]:
    return json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))


def place_of(path: Path_, root: str = "report") -> str:
    place = root
    for part in path:
        place += f"[{part}]" if isinstance(part, int) else f".{part}"
    return place


def closed_places(shape: Any, node: Any, path: Path_ = ()) -> Iterator[Path_]:
    """The path of every mapping in `node` that `shape` fixes the keys of."""
    if isinstance(shape, Nullable):
        if node is not None:
            yield from closed_places(shape.shape, node, path)
    elif isinstance(shape, Keys):
        yield path
        for key, sub in shape.shapes.items():
            yield from closed_places(sub, node[key], (*path, key))
    elif isinstance(shape, Table):
        for key, item in node.items():
            yield from closed_places(shape.value, item, (*path, key))
    elif isinstance(shape, Seq):
        for position, item in enumerate(node):
            yield from closed_places(shape.item, item, (*path, position))


def _walk(document: Any, path: Path_) -> Any:
    for part in path:
        document = document[part]
    return document


def with_key(
    document: dict[str, Any], path: Path_, key: str, value: Any
) -> dict[str, Any]:
    """A copy of `document` with `key` set to `value` in the mapping at `path`."""
    copied = copy.deepcopy(document)
    _walk(copied, path)[key] = value
    return copied


def without_key(document: dict[str, Any], path: Path_, key: str) -> dict[str, Any]:
    copied = copy.deepcopy(document)
    del _walk(copied, path)[key]
    return copied


def with_value(document: dict[str, Any], path: Path_, value: Any) -> dict[str, Any]:
    """A copy of `document` whose value at `path` (not empty) is `value`."""
    copied = copy.deepcopy(document)
    _walk(copied, path[:-1])[path[-1]] = value
    return copied


def dumps(document: Any) -> bytes:
    """The canonical bytes of a document: sorted keys, two-space indent, one final line feed."""
    return (
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode()
