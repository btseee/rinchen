"""Tests for `rinchen.export.eole_ct2`: the converter call that moved out of
`rinchen.script_conv.export` in D10, so that `rinchen.train` can export a teacher without
importing `script_conv`. The behaviour of the three functions is pinned by D5's own suite
(`test_script_conv_export.py`), which runs unchanged over the names `script_conv.export`
re-exports; what is new here is that the names ARE the moved ones and that the new module
keeps to its layer.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from rinchen.errors import UpstreamError
from rinchen.export import eole_ct2
from rinchen.script_conv import export as script_conv_export

MOVED = ("export_ct2", "export_environment", "export_log_path")
FORBIDDEN_LAYERS = (
    "rinchen.train",
    "rinchen.synth",
    "rinchen.cli",
    "rinchen.script_conv",
)


def imported_modules(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


@pytest.mark.parametrize("name", MOVED)
def test_the_converter_functions_of_the_script_converter_are_the_moved_ones(
    name: str,
) -> None:
    moved = getattr(eole_ct2, name)

    assert getattr(script_conv_export, name) is moved
    assert moved.__module__ == "rinchen.export.eole_ct2"


def test_the_script_converter_module_still_defines_the_parity_check() -> None:
    from rinchen.script_conv.export import ParityReport, parity

    assert parity.__module__ == "rinchen.script_conv.export"
    assert ParityReport.__module__ == "rinchen.script_conv.export"


def test_the_script_converter_module_lists_what_it_exports() -> None:
    assert sorted(script_conv_export.__all__) == [
        "ParityReport",
        "export_ct2",
        "export_environment",
        "export_log_path",
        "parity",
    ]


def test_the_export_package_imports_nothing_of_the_layers_above_it() -> None:
    imported = imported_modules(Path(eole_ct2.__file__))

    for module in imported:
        assert not module.startswith(FORBIDDEN_LAYERS), module
    # The converter runs as a subprocess: nothing here needs CTranslate2 in process.
    assert "ctranslate2" not in imported
    assert "rinchen.utils.logged_process" in imported


def test_the_export_package_holds_no_other_module() -> None:
    package = Path(eole_ct2.__file__).parent

    assert sorted(path.name for path in package.glob("*.py")) == [
        "__init__.py",
        "eole_ct2.py",
    ]
    assert (package / "__init__.py").read_text(encoding="utf-8") == ""


def exits_zero_without_a_model(
    monkeypatch: pytest.MonkeyPatch, built: list[Path]
) -> None:
    """Replace the converter's process by one that exits zero and leaves no `model.bin` in
    the directory it was told to write to (recorded in `built`)."""

    def run(command: Any, *, log_path: Path, **options: Any) -> None:
        built.append(Path(command[command.index("--output_dir") + 1]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("converter log\n", encoding="utf-8")

    monkeypatch.setattr(eole_ct2, "run_logged", run)


def test_a_missing_model_is_reported_in_the_output_directory_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exits_zero_without_a_model(monkeypatch, [])

    with pytest.raises(UpstreamError) as raised:
        eole_ct2.export_ct2(tmp_path / "ckpt", tmp_path / "ct2")

    assert f"left no model.bin in {tmp_path / 'ct2'}, or an empty one" in str(
        raised.value
    )


def test_a_missing_model_is_looked_for_in_the_output_directory_and_reported_in_the_one_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Path] = []
    exits_zero_without_a_model(monkeypatch, built)
    private = tmp_path / ".ct2.private" / "ct2"

    with pytest.raises(UpstreamError) as raised:
        eole_ct2.export_ct2(
            tmp_path / "ckpt",
            private,
            log_path=tmp_path / "published.export.log",
            reported_dir=tmp_path / "published",
        )

    message = str(raised.value)
    assert built == [private]
    assert f"left no model.bin in {tmp_path / 'published'}, or an empty one" in message
    assert ".ct2.private" not in message
