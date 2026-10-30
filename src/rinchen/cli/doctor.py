"""`rinchen doctor`: environment checks with clear pass/fail/skip results."""

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=False)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass(slots=True, frozen=True)
class DoctorResult:
    check_id: str
    status: str  # "pass" | "fail" | "skip"
    message: str


def check_py() -> DoctorResult:
    if sys.version_info[:2] == (3, 14):
        return DoctorResult("D-PY", "pass", f"CPython {sys.version.split()[0]}")
    return DoctorResult(
        "D-PY", "fail", f"expected CPython 3.14.x, got {sys.version.split()[0]}"
    )


def _parse_pinned_package_names(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    names = []
    for entry in data["project"]["dependencies"]:
        name = entry.split("=")[0].split(">")[0].split("<")[0].split("[")[0].strip()
        names.append(name)
    return names


def check_wheels() -> DoctorResult:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    names = _parse_pinned_package_names(pyproject_path)
    failures: list[str] = []
    versions: list[str] = []
    for name in names:
        try:
            version = importlib.metadata.version(name)
            versions.append(f"{name}=={version}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(name)
    if failures:
        return DoctorResult("D-WHEELS", "fail", f"not installed: {', '.join(failures)}")
    return DoctorResult("D-WHEELS", "pass", "; ".join(versions))


def check_eole_py() -> DoctorResult:
    try:
        importlib.import_module("eole")
    except Exception as exc:  # noqa: BLE001 -- doctor check: any import failure is a fail
        return DoctorResult("D-EOLE-PY", "fail", f"import eole failed: {exc}")
    return DoctorResult("D-EOLE-PY", "pass", "import eole succeeded")


def _run_help(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return result.returncode == 0, (result.stdout + result.stderr)[-500:]


def check_eole_cli() -> DoctorResult:
    subcommands = ["build_vocab", "train", "predict", "convert"]
    failures = []
    for sub in subcommands:
        ok, output = _run_help([sys.executable, "-m", "eole.bin.main", sub, "--help"])
        if not ok:
            failures.append(f"{sub}: {output}")
    if failures:
        return DoctorResult("D-EOLE-CLI", "fail", "; ".join(failures))
    note = (
        " (NOTE: 'eole model average' from the master plan does not exist in"
        " installed eole 0.6.4's CLI -- top-level subcommands are"
        " {convert,build_vocab,predict,serve,train,tools}, no 'model' group;"
        " see docs/VERIFY.md)"
    )
    return DoctorResult(
        "D-EOLE-CLI", "pass", f"{len(subcommands)} subcommands parsed --help{note}"
    )


def check_ct2_conv() -> DoctorResult:
    ok, output = _run_help(
        [sys.executable, "-m", "ctranslate2.converters.eole_ct2", "--help"]
    )
    if ok:
        return DoctorResult(
            "D-CT2-CONV", "pass", "python -m ctranslate2.converters.eole_ct2 --help"
        )
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    matches = [ep.name for ep in entry_points if "eole" in ep.name.lower()]
    if matches:
        return DoctorResult("D-CT2-CONV", "pass", f"found console scripts: {matches}")
    return DoctorResult(
        "D-CT2-CONV",
        "fail",
        f"module --help failed ({output}) and no eole console script found",
    )


def check_ct2_import() -> DoctorResult:
    torch_was_loaded_before = "torch" in sys.modules
    try:
        importlib.import_module("ctranslate2")
    except Exception as exc:  # noqa: BLE001 -- doctor check: any import failure is a fail
        return DoctorResult("D-CT2-IMPORT", "fail", f"import ctranslate2 failed: {exc}")
    torch_loaded_as_side_effect = not torch_was_loaded_before and "torch" in sys.modules
    note = (
        " (torch imported as a side effect, see CTranslate2 issue #2079)"
        if torch_loaded_as_side_effect
        else ""
    )
    return DoctorResult("D-CT2-IMPORT", "pass", f"import ctranslate2 succeeded{note}")


def check_torch_cuda() -> DoctorResult:
    try:
        import torch
    except ImportError as exc:
        return DoctorResult("D-TORCH-CUDA", "skip", f"torch not importable: {exc}")
    if torch.cuda.is_available():
        return DoctorResult(
            "D-TORCH-CUDA", "pass", f"CUDA available: {torch.cuda.get_device_name(0)}"
        )
    return DoctorResult("D-TORCH-CUDA", "skip", "no GPU available")


ALL_CHECKS = [
    check_py,
    check_wheels,
    check_eole_py,
    check_eole_cli,
    check_ct2_conv,
    check_ct2_import,
    check_torch_cuda,
]


def run_all() -> list[DoctorResult]:
    return [check() for check in ALL_CHECKS]


@app.callback(invoke_without_command=True)
def main() -> None:
    """Run every doctor check and print a summary; exit non-zero on any failure."""
    results = run_all()
    failed = False
    for result in results:
        marker = {"pass": "OK", "fail": "FAIL", "skip": "SKIP"}[result.status]
        typer.echo(f"[{marker}] {result.check_id}: {result.message}")
        if result.status == "fail":
            failed = True
    raise typer.Exit(code=1 if failed else 0)
