from __future__ import annotations

from typer.testing import CliRunner

from rinchen.cli import data as data_cli
from rinchen.cli.main import app

runner = CliRunner()


def test_data_fetch_tiny_profile_works_fully_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cli, "DATA_RAW_DIR", tmp_path)

    result = runner.invoke(app, ["data", "fetch", "--profile", "tiny", "--all"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "opus-api-sample" / "opus_api_mn_en.json").is_file()
    assert (tmp_path / "warc-sample" / "sample.warc.gz").is_file()
    assert (tmp_path / "news-html-sample" / "mn_news.html").is_file()
    assert (tmp_path / "moses-zip-sample" / "moses_mn_en.zip").is_file()


def test_data_ls_tiny_profile_lists_all_four_fixtures():
    result = runner.invoke(app, ["data", "ls", "--profile", "tiny"])

    assert result.exit_code == 0, result.output
    assert "opus-api-sample" in result.output
    assert "warc-sample" in result.output
    assert "news-html-sample" in result.output
    assert "moses-zip-sample" in result.output


def test_data_fetch_requires_source_or_all():
    result = runner.invoke(app, ["data", "fetch", "--profile", "tiny"])

    assert result.exit_code == 2
