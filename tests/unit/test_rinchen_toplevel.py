"""`import rinchen` stays cheap and `rinchen.Translator` is there when asked for (D11, ruling
T14): a lazy module attribute (PEP 562). Every check of `sys.modules` runs in a fresh
interpreter on this tree, because the test process has imported all of it already."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import rinchen

ROOT = Path(rinchen.__file__).resolve().parents[2]
HEAVY = ("ctranslate2", "sentencepiece", "eole", "torch")


def run(code: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    prelude = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(ROOT / "src")!r})
        import rinchen
        assert rinchen.__file__.startswith({str(ROOT / "src")!r}), rinchen.__file__
        """
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=timeout,
        check=False,
    )


def test_importing_the_package_loads_no_other_module_of_it_and_no_heavy_library() -> (
    None
):
    result = run(
        f"""
        print(sorted(m for m in sys.modules if m.startswith("rinchen.")))
        print([m for m in {HEAVY!r} if m in sys.modules])
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["[]", "[]"]


def test_the_translator_resolves_on_first_use_and_the_libraries_wait_for_a_model() -> (
    None
):
    result = run(
        f"""
        assert "rinchen.infer" not in sys.modules
        translator = rinchen.Translator
        from rinchen.infer.translator import Translator
        assert translator is Translator
        assert "rinchen.infer.engine" in sys.modules
        print([m for m in {HEAVY!r} if m in sys.modules])
        print(rinchen.Translator is Translator)
        """
    )

    assert result.returncode == 0, result.stderr
    # `ctranslate2` and `sentencepiece` are imported when a model is opened, not before.
    assert result.stdout.splitlines() == ["[]", "True"]


def test_from_import_gets_the_translator() -> None:
    result = run(
        """
        from rinchen import Translator
        from rinchen.infer.translator import Translator as Facade
        print(Translator is Facade, Translator.__name__)
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True Translator"


def test_the_listing_of_the_package_names_the_translator() -> None:
    assert "Translator" in dir(rinchen)
    assert rinchen.__all__ == ["Translator"]


def test_an_unknown_name_is_an_attribute_error_that_names_it() -> None:
    with pytest.raises(AttributeError, match="no_such_thing"):
        rinchen.no_such_thing  # noqa: B018

    assert not hasattr(rinchen, "Translater")


def test_a_failed_lookup_loads_nothing() -> None:
    result = run(
        """
        try:
            rinchen.Nope
        except AttributeError:
            pass
        print("rinchen.infer" in sys.modules)
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_mypy_sees_the_translator_as_the_class_it_is() -> None:
    code = "from rinchen import Translator\nreveal_type(Translator)\nTranslator()\n"

    result = subprocess.run(
        [sys.executable, "-I", "-m", "mypy", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "MYPYPATH": str(ROOT / "src")},
        timeout=600,
        check=False,
        cwd=ROOT,
    )

    # The lazy lookup must not turn the name into `Any`: the class is typed, so a call
    # without its directory is the one error mypy finds.
    errors = [line for line in result.stdout.splitlines() if ": error:" in line]
    assert len(errors) == 1, result.stdout + result.stderr
    assert 'Missing positional argument "directory"' in errors[0]
    assert "rinchen.infer.translator.Translator" in result.stdout
