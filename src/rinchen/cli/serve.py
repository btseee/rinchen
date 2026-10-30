"""Stub CLI sub-app for `rinchen serve` -- real commands land in D11 (Export, quantization, serving)."""
import typer

app = typer.Typer(no_args_is_help=True)
