from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rinchen.cli.main import app

runner = CliRunner()

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "annot"


def test_init_creates_db_and_seeds_annotators(tmp_path):
    db_path = tmp_path / "test.sqlite"
    result = runner.invoke(
        app, ["annotate", "init", "--db", str(db_path), "--annotators", "alice,bob"]
    )
    assert result.exit_code == 0, result.output
    assert db_path.exists()


def test_import_reports_count(tmp_path):
    db_path = tmp_path / "test.sqlite"
    runner.invoke(
        app, ["annotate", "init", "--db", str(db_path), "--annotators", "alice,bob"]
    )
    result = runner.invoke(
        app,
        [
            "annotate",
            "import",
            "--db",
            str(db_path),
            "--file",
            str(FIXTURES_DIR / "import_sample.jsonl"),
            "--set",
            "demo",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3" in result.output


def test_export_unknown_format_is_usage_error(tmp_path):
    db_path = tmp_path / "test.sqlite"
    runner.invoke(
        app, ["annotate", "init", "--db", str(db_path), "--annotators", "alice"]
    )
    result = runner.invoke(
        app,
        [
            "annotate",
            "export",
            "--db",
            str(db_path),
            "--set",
            "demo",
            "--format",
            "xml",
            "--out",
            str(db_path.parent / "out.xml"),
        ],
    )
    assert result.exit_code == 2


def test_agreement_reports_na_on_empty_set(tmp_path):
    db_path = tmp_path / "test.sqlite"
    runner.invoke(
        app, ["annotate", "init", "--db", str(db_path), "--annotators", "alice"]
    )
    result = runner.invoke(
        app, ["annotate", "agreement", "--db", str(db_path), "--set", "demo"]
    )
    assert result.exit_code == 0, result.output
    assert "N/A" in result.output
