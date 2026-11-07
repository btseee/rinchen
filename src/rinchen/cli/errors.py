"""How the `rinchen` commands turn a `RinchenError` into an exit code and an
`error: <message>` line on stderr. Shared by every command that does real work,
so that the codes of the master plan are defined once."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import NoReturn

import typer

from rinchen.errors import ConfigError, DataError, RinchenError, UpstreamError

CONFIG_ERROR_EXIT_CODE = 2
DATA_ERROR_EXIT_CODE = 3
UPSTREAM_ERROR_EXIT_CODE = 4
# A RinchenError of any other kind, or an output-side OSError, is a plain
# failure.
FAILURE_EXIT_CODE = 1

_EXIT_CODE_BY_ERROR_TYPE: dict[type[RinchenError], int] = {
    ConfigError: CONFIG_ERROR_EXIT_CODE,
    DataError: DATA_ERROR_EXIT_CODE,
    UpstreamError: UPSTREAM_ERROR_EXIT_CODE,
}


def exit_with_error(message: str, exit_code: int) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=exit_code)


@contextmanager
def rinchen_errors_as_exit_codes(
    *, message_of: Callable[[RinchenError], str] = str
) -> Iterator[None]:
    """Exit with the code of a `RinchenError` raised in the block, printing
    `message_of(error)`. Other exceptions pass through."""
    try:
        yield
    except RinchenError as exc:
        # The error classes are siblings, so the order of this lookup does not
        # matter; a subclass would have to be listed before its parent.
        exit_code = next(
            (
                code
                for error_type, code in _EXIT_CODE_BY_ERROR_TYPE.items()
                if isinstance(exc, error_type)
            ),
            FAILURE_EXIT_CODE,
        )
        exit_with_error(message_of(exc), exit_code)
