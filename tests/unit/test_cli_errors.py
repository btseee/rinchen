"""The CLI error mapping shared by `rinchen spm`, `filter` and `mine`: each
`RinchenError` kind becomes its documented exit code and an `error: <msg>` line
on stderr; everything else passes through untouched."""

from __future__ import annotations

import pytest
import typer

from rinchen.cli.errors import (
    CONFIG_ERROR_EXIT_CODE,
    DATA_ERROR_EXIT_CODE,
    FAILURE_EXIT_CODE,
    UPSTREAM_ERROR_EXIT_CODE,
    exit_with_error,
    rinchen_errors_as_exit_codes,
)
from rinchen.errors import (
    ConfigError,
    DataError,
    ParityError,
    RinchenError,
    UpstreamError,
)


class SpecialConfigError(ConfigError):
    pass


def test_the_exit_codes_are_the_master_plans() -> None:
    assert (
        CONFIG_ERROR_EXIT_CODE,
        DATA_ERROR_EXIT_CODE,
        UPSTREAM_ERROR_EXIT_CODE,
        FAILURE_EXIT_CODE,
    ) == (2, 3, 4, 1)


def test_exit_with_error_prints_the_message_to_stderr_and_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as raised:
        exit_with_error("something broke", 7)

    assert raised.value.exit_code == 7
    captured = capsys.readouterr()
    assert captured.err == "error: something broke\n"
    assert captured.out == ""


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (ConfigError("bad config"), 2),
        (DataError("bad data"), 3),
        (UpstreamError("bad upstream"), 4),
        (ParityError("bad parity"), 1),
        (RinchenError("bad"), 1),
        (SpecialConfigError("subclass of a mapped kind"), 2),
    ],
)
def test_each_rinchen_error_kind_maps_to_its_exit_code(
    error: RinchenError, exit_code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(typer.Exit) as raised, rinchen_errors_as_exit_codes():
        raise error

    assert raised.value.exit_code == exit_code
    assert capsys.readouterr().err == f"error: {error}\n"


def test_a_block_without_an_error_is_left_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with rinchen_errors_as_exit_codes():
        pass

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_an_error_outside_the_rinchen_family_passes_through() -> None:
    with pytest.raises(ValueError, match="not ours"), rinchen_errors_as_exit_codes():
        raise ValueError("not ours")


def test_a_message_hook_replaces_the_text_that_is_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        pytest.raises(typer.Exit) as raised,
        rinchen_errors_as_exit_codes(message_of=lambda exc: f"[{exc}]"),
    ):
        raise DataError("detail")

    assert raised.value.exit_code == 3
    assert capsys.readouterr().err == "error: [detail]\n"
