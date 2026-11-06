"""`rinchen spm`: train a SentencePiece model and check it against the project
invariants. Presentation only; the work happens in `rinchen.spm`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from rinchen.config.spm import load_spm_config
from rinchen.errors import ConfigError, DataError, RinchenError, UpstreamError
from rinchen.spm.check import SpmCheckReport, check_spm
from rinchen.spm.corpus import discover_corpus
from rinchen.spm.train import train_spm

app = typer.Typer(no_args_is_help=True)

# The master plan's CLI exit codes. A RinchenError of any other kind, or an
# output-side OSError, is a plain failure.
_EXIT_CODE_BY_ERROR_TYPE: dict[type[RinchenError], int] = {
    ConfigError: 2,
    DataError: 3,
    UpstreamError: 4,
}
_FAILURE_EXIT_CODE = 1
_FAILED_CHECK_EXIT_CODE = _EXIT_CODE_BY_ERROR_TYPE[DataError]


def _exit_with_error(message: str, exit_code: int) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=exit_code)


@contextmanager
def _rinchen_errors_as_exit_codes() -> Iterator[None]:
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
            _FAILURE_EXIT_CODE,
        )
        _exit_with_error(str(exc), exit_code)


def _report_lines(report: SpmCheckReport) -> list[str]:
    byte_fallback = "on" if report.byte_fallback_on else "off"
    return [
        *(f"{entry.lang}\t{entry.coverage:.4f}" for entry in report.coverage),
        f"byte_fallback: {byte_fallback}",
        *(f"non-atomic symbol: {symbol}" for symbol in report.non_atomic_symbols),
        "ok" if report.ok else "failed",
    ]


@app.command()
def train(
    config: Annotated[
        Path, typer.Option("--config", help="SentencePiece config, e.g. tiny_1k.yaml.")
    ],
) -> None:
    with _rinchen_errors_as_exit_codes():
        cfg = load_spm_config(config)
        try:
            model_path = train_spm(cfg)
        except OSError as exc:
            _exit_with_error(
                f"cannot write SentencePiece outputs for {config}: {exc}",
                _FAILURE_EXIT_CODE,
            )
    typer.echo(f"trained {model_path}")


@app.command()
def check(
    model: Annotated[Path, typer.Option("--model", help="Trained .model file.")],
    corpus_dir: Annotated[
        Path,
        typer.Option("--corpus-dir", help="Directory of <language code>.txt files."),
    ],
    sample_lines: Annotated[
        int,
        typer.Option(
            "--sample-lines",
            min=1,
            help="Lines per language to measure coverage on.",
        ),
    ] = 1000,
) -> None:
    with _rinchen_errors_as_exit_codes():
        report = check_spm(
            model, discover_corpus(corpus_dir), sample_lines=sample_lines
        )
    for line in _report_lines(report):
        typer.echo(line)
    if not report.ok:
        raise typer.Exit(code=_FAILED_CHECK_EXIT_CODE)
