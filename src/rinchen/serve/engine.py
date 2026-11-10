"""What the service reads of an engine (D11, T15): `ServingEngine`.

`rinchen.infer.engine.Engine` has this shape, and so does the fake engine of the tests. It is
public because the command that builds the service (`rinchen.cli.serve`) names it as well as the
service does; the rest of `rinchen.serve._translate` stays private to the package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rinchen.infer.engine_types import (
    DecodeOptions,
    DocumentResult,
    ModelInfo,
    Row,
    ScriptName,
    Translation,
)


class ServingEngine(Protocol):
    """What the service reads of an engine."""

    @property
    def info(self) -> ModelInfo: ...

    def translate(
        self, rows: Sequence[Row], options: DecodeOptions | None = None
    ) -> list[Translation]: ...

    def translate_documents(
        self,
        documents: Sequence[str],
        src: str | None,
        tgt: str,
        options: DecodeOptions | None = None,
        *,
        script: ScriptName | None = None,
    ) -> list[DocumentResult]: ...

    def warmup(self) -> None: ...

    def counters(self) -> dict[str, int]: ...

    def close(self) -> None: ...
