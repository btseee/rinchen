"""The order in which `rinchen.script_conv.train.predict` checks and acts.

D5 first refused every bad call, then made the output directory, and only then removed
the output of an earlier run: a call that fails before Eole runs leaves the old
`out_file` alone, and one that fails while it runs (or after) leaves none. The Eole
machinery moved to `rinchen.train.eole_run` after D5 was done; these tests pin the
order from the outside, with `subprocess.Popen` replaced by a recorder (as in
`test_script_conv_train.py`), so they hold for either shape of the code.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from rinchen.errors import UpstreamError
from rinchen.script_conv.char_model import TO_SCRIPT_TAG, train_char_model
from rinchen.script_conv.train import predict
from tests.support.fake_process import FakeProcess

EARLIER = "left from an earlier run\n"


class FakeEole:
    """Stands in for `subprocess.Popen`: records each launch, and when it exits with
    `exit_code`, leaves the raw output (one token line per source line, the tag
    dropped) like the real tool."""

    def __init__(self) -> None:
        self.launches: list[list[str]] = []
        self.exit_code = 0

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> FakeProcess:
        self.launches.append(list(argv))
        if self.exit_code == 0:
            cfg = yaml.safe_load(Path(argv[3]).read_text(encoding="utf-8"))
            source = Path(cfg["src"]).read_text(encoding="utf-8").splitlines()
            Path(cfg["output"]).write_text(
                "".join(" ".join(line.split(" ")[1:]) + "\n" for line in source),
                encoding="utf-8",
            )
        return FakeProcess(self.exit_code, [])


@pytest.fixture
def fake_eole(monkeypatch: pytest.MonkeyPatch) -> FakeEole:
    fake = FakeEole()
    monkeypatch.setattr(subprocess, "Popen", fake)
    return fake


@pytest.fixture(scope="module")
def spm_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    prefix = tmp_path_factory.mktemp("order_model") / "chars"
    return train_char_model(["abc", "cab"], prefix, vocab_size=64)


@pytest.fixture
def stale_out_file(tmp_path: Path) -> Path:
    out_file = tmp_path / "out" / "pred.tgt"
    out_file.parent.mkdir()
    out_file.write_text(EARLIER, encoding="utf-8")
    return out_file


def _predict(tmp_path: Path, spm_model: Path, out_file: Path, **keywords: Any) -> Path:
    src_file = tmp_path / "words.src"
    src_file.write_text("abc\ncab\n", encoding="utf-8")
    return predict(
        tmp_path / "ckpt",
        src_file,
        tag=TO_SCRIPT_TAG,
        spm_model=spm_model,
        out_file=out_file,
        **keywords,
    )


def _hide_eole(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bin_dir = tmp_path / "no_eole"
    bin_dir.mkdir()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))


def test_predict_keeps_the_output_of_an_earlier_run_when_eole_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    fake_eole: FakeEole,
    tmp_path: Path,
    spm_model: Path,
    stale_out_file: Path,
) -> None:
    _hide_eole(monkeypatch, tmp_path)

    with pytest.raises(UpstreamError, match="the `eole` command is not installed"):
        _predict(tmp_path, spm_model, stale_out_file)

    assert stale_out_file.read_text(encoding="utf-8") == EARLIER
    assert fake_eole.launches == []


@pytest.mark.parametrize("threads", [0, -1])
def test_predict_keeps_the_output_of_an_earlier_run_when_the_thread_count_is_refused(
    fake_eole: FakeEole,
    tmp_path: Path,
    spm_model: Path,
    stale_out_file: Path,
    threads: int,
) -> None:
    with pytest.raises(ValueError, match="threads must be at least 1"):
        _predict(tmp_path, spm_model, stale_out_file, threads=threads)

    assert stale_out_file.read_text(encoding="utf-8") == EARLIER
    assert fake_eole.launches == []


def test_predict_reports_a_missing_eole_before_a_bad_thread_count(
    monkeypatch: pytest.MonkeyPatch,
    fake_eole: FakeEole,
    tmp_path: Path,
    spm_model: Path,
    stale_out_file: Path,
) -> None:
    _hide_eole(monkeypatch, tmp_path)

    with pytest.raises(UpstreamError, match="the `eole` command is not installed"):
        _predict(tmp_path, spm_model, stale_out_file, threads=0)


def test_predict_removes_the_output_of_an_earlier_run_when_eole_fails(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path, stale_out_file: Path
) -> None:
    fake_eole.exit_code = 1

    with pytest.raises(UpstreamError, match="eole predict failed"):
        _predict(tmp_path, spm_model, stale_out_file)

    assert not stale_out_file.exists()


def test_predict_reports_a_file_where_the_output_directory_should_be_as_file_exists(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    blocker = tmp_path / "out"
    blocker.write_text("a regular file", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _predict(tmp_path, spm_model, blocker / "pred.tgt")

    assert fake_eole.launches == []
