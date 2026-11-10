"""`rinchen eval suite inspect` (D13, Task 4): what the command prints about a suite file, line by
line, and the exit code for a file that is not a suite (3). The file is hashed before it is read and
the bytes that were hashed are the bytes that are parsed."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from rinchen.cli import eval_suite
from rinchen.eval.suite_rules import MAX_SUITE_BYTES
from tests.support.eval_suite_cli import (
    REVISION,
    FakeLoadDataset,
    fixture_rows,
    install_fake_load_dataset,
    invoke,
    make_workspace,
    sha256_of,
    write_config,
)
from tests.support.launch_cli import assert_one_error
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark

pytestmark = pytest.mark.usefixtures("no_network")

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "eval" / "golden_suite.json"
GOLDEN_SHA256 = "b707518d39133bd206092e6df28bf0bcf1e42c776ad5c16900995bf783ebc761"


def inspect_args(suite: Path) -> list[str]:
    return ["eval", "suite", "inspect", "--suite", str(suite)]


def test_the_golden_suite_prints_exactly_its_description() -> None:
    result = invoke(inspect_args(GOLDEN))

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "name: golden",
        "kind: fixture",
        "split: devtest",
        "purpose: report",
        f"sha256: {GOLDEN_SHA256}",
        "rows: eng_Latn 10, khk_Cyrl 10",
        "directions: 2",
        "  khk_Cyrl-eng_Latn",
        "  eng_Latn-khk_Cyrl",
    ]
    assert result.stderr == ""


def test_the_digest_printed_is_the_one_of_the_file() -> None:
    assert sha256_of(GOLDEN) == GOLDEN_SHA256


def test_a_dev_suite_says_its_purpose_is_selection(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    assert (
        invoke(workspace.build_args(split="dev", out_dir=tmp_path / "out")).exit_code
        == 0
    )

    result = invoke(inspect_args(tmp_path / "out" / "flores_plus-dev.json"))

    assert "split: dev" in result.stdout.splitlines()
    assert "purpose: selection" in result.stdout.splitlines()


def test_a_suite_built_from_loose_shards_prints_no_revision_line(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    assert invoke(workspace.build_args()).exit_code == 0

    lines = invoke(inspect_args(workspace.suite)).stdout.splitlines()

    assert not any(line.startswith("revision:") or line == "unpinned" for line in lines)


def test_the_languages_the_source_lacks_are_printed_when_there_are_any(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    assert invoke(workspace.build_args()).exit_code == 0

    lines = invoke(inspect_args(workspace.suite)).stdout.splitlines()

    assert "rows: eng_Latn 4, khk_Cyrl 4, rus_Cyrl 4" in lines
    assert "directions: 4" in lines
    assert (
        "missing: deu_Latn, jpn_Jpan, kaz_Cyrl, kor_Hang, tur_Latn, zho_Hans" in lines
    )


def fetched_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> Path:
    workspace = make_workspace(tmp_path)
    config = write_config(tmp_path / "suites.yaml")
    install_fake_load_dataset(monkeypatch, FakeLoadDataset(fixture_rows()))
    fetched = tmp_path / "fetched"
    fetch = invoke(
        [
            "eval",
            "suite",
            "fetch",
            "--benchmark",
            "flores_plus",
            "--split",
            "devtest",
            "--out-dir",
            str(fetched),
            "--config",
            str(config),
            *extra,
        ]
    )
    assert fetch.exit_code == 0, fetch.output
    build = invoke(
        workspace.build_args("--config", str(config), source=fetched, split="devtest")
    )
    assert build.exit_code == 0, build.output
    return workspace.suite


def test_a_pinned_suite_prints_its_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = fetched_suite(tmp_path, monkeypatch, "--revision", REVISION)

    lines = invoke(inspect_args(suite)).stdout.splitlines()

    assert f"revision: {REVISION}" in lines
    assert "unpinned" not in lines


def test_a_suite_of_a_fetch_without_a_pin_says_unpinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = fetched_suite(tmp_path, monkeypatch, "--unpinned")

    lines = invoke(inspect_args(suite)).stdout.splitlines()

    assert "unpinned" in lines
    assert not any(line.startswith("revision:") for line in lines)


@pytest.mark.parametrize(
    "content",
    [b"", b"not json", b"[]", b'{"schema": "rinchen.eval.suite/1"}', b"\xff\xfe"],
    ids=["empty", "not-json", "a-list", "missing-fields", "not-utf-8"],
)
def test_a_file_that_is_not_a_suite_is_exit_3_and_prints_nothing(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "suite.json"
    path.write_bytes(content)

    result = invoke(inspect_args(path))

    assert_one_error(result, 3, "not an evaluation suite")
    assert result.stdout == ""


def test_a_file_that_is_missing_is_exit_3(tmp_path: Path) -> None:
    result = invoke(inspect_args(tmp_path / "nothing.json"))

    assert_one_error(result, 3, "")
    assert result.stdout == ""


def test_a_file_over_the_cap_is_refused_without_reading_it_into_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.json"
    with path.open("wb") as handle:
        handle.truncate(MAX_SUITE_BYTES + 1)
    reads: list[Path] = []
    real_read_bytes = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        reads.append(self)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    result = invoke(inspect_args(path))

    assert_one_error(result, 3, "not an evaluation suite")
    assert "64 MiB" in result.stderr
    assert path not in reads


def test_the_file_is_hashed_before_it_is_read_and_the_hashed_bytes_are_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = shutil.copyfile(GOLDEN, tmp_path / "golden.json")
    real_hash = eval_suite.hash_input

    def hashing(file: Path, **options: Any) -> Any:
        entry = real_hash(file, **options)
        file.write_bytes(file.read_bytes().replace(b"golden", b"silver"))
        return entry

    monkeypatch.setattr(eval_suite, "hash_input", hashing)

    result = invoke(inspect_args(path))

    assert_one_error(result, 2, "")
    assert "changed after it was hashed" in result.stderr
    assert result.stdout == ""


def test_inspect_prints_no_text_of_the_suite() -> None:
    result = invoke(inspect_args(GOLDEN))

    assert "mat" not in result.output
    assert "муур" not in result.output
