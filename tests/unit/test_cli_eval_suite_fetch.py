"""`rinchen eval suite fetch` (D13, Task 4): the only step of D13 that is meant to reach the network,
and a step no test runs for real. The real `rinchen.acquire.hf.fetch` runs here with
`datasets.load_dataset` replaced by a recording fake, under the `no_network` guard, so what the
tests show is which calls the command makes (one per configured language, with the language, the
split, `streaming=True` and the revision), which files it writes and what the manifest says. An
unpinned revision is refused unless the operator says `--unpinned`; a token is never printed, read
from a file or written."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from rinchen.config.eval_suites import DEFAULT_EVAL_SUITES_PATH

from tests.support.distill_cli import write_foreign_manifest
from tests.support.eval_suite_cli import (
    LANGUAGES,
    MANIFEST_FILE,
    REVISION,
    SHARD,
    FakeLoadDataset,
    fixture_rows,
    install_fake_load_dataset,
    invoke,
    read_manifest,
    sha256_of,
    shard_rows,
    write_config,
)
from tests.support.launch_cli import assert_one_error, snapshot
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.robust_cli import fail_replace, record_replaces

pytestmark = pytest.mark.usefixtures("no_network")

NINE = (
    "khk_Cyrl",
    "eng_Latn",
    "rus_Cyrl",
    "zho_Hans",
    "kor_Hang",
    "jpn_Jpan",
    "deu_Latn",
    "kaz_Cyrl",
    "tur_Latn",
)
DATASET = "openlanguagedata/flores_plus"


def three_rows(languages: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    return {language: shard_rows(language)[:3] for language in languages}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLoadDataset:
    return install_fake_load_dataset(monkeypatch, FakeLoadDataset(three_rows(NINE)))


def fetch_args(out_dir: Path, *extra: str, split: str = "dev") -> list[str]:
    return [
        "eval",
        "suite",
        "fetch",
        "--benchmark",
        "flores_plus",
        "--split",
        split,
        "--out-dir",
        str(out_dir),
        *extra,
    ]


# ---- a pinned fetch -----------------------------------------------------------------------


def test_a_pinned_fetch_makes_one_call_per_configured_language_with_the_revision(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    result = invoke(fetch_args(tmp_path / "out", "--revision", REVISION))

    assert result.exit_code == 0, result.output
    assert [call.as_tuple() for call in fake.calls] == [
        (DATASET, language, "dev", True, {"revision": REVISION}) for language in NINE
    ]


def test_the_shards_are_written_under_the_split_and_the_language(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    assert invoke(fetch_args(out_dir, "--revision", REVISION)).exit_code == 0

    for language in NINE:
        shard = out_dir / "dev" / language / SHARD
        assert pl.read_parquet(shard).to_dicts() == shard_rows(language)[:3]


def test_the_manifest_vouches_for_every_shard_and_says_what_was_fetched(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    assert invoke(fetch_args(out_dir, "--revision", REVISION)).exit_code == 0

    manifest = read_manifest(out_dir)

    assert manifest["stage"] == "eval_suite_fetch"
    assert manifest["seed"] == -1
    assert [
        (item["path"], item["sha256"], item["rows"]) for item in manifest["outputs"]
    ] == [
        (
            str(out_dir / "dev" / language / SHARD),
            sha256_of(out_dir / "dev" / language / SHARD),
            3,
        )
        for language in NINE
    ]
    assert [item["path"] for item in manifest["inputs"]] == [
        str(DEFAULT_EVAL_SUITES_PATH)
    ]
    assert manifest["inputs"][0]["sha256"] == sha256_of(DEFAULT_EVAL_SUITES_PATH)
    assert manifest["status"] == {
        "revision": REVISION,
        "benchmark": "flores_plus",
        "split": "dev",
        "dataset": DATASET,
    }
    assert manifest["metrics"] == {f"rows.{language}": 3.0 for language in NINE}
    assert manifest["tool_versions"] == {
        package: importlib.metadata.version(package)
        for package in ("datasets", "huggingface_hub", "polars")
    }


def test_the_manifest_is_written_last(
    fake: FakeLoadDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = record_replaces(monkeypatch)

    assert invoke(fetch_args(tmp_path / "out", "--revision", REVISION)).exit_code == 0

    assert names[-1] == MANIFEST_FILE
    assert names.count(MANIFEST_FILE) == 1


def test_the_command_prints_what_it_fetched(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"

    result = invoke(fetch_args(out_dir, "--revision", REVISION))

    assert result.stdout.splitlines() == [
        "benchmark: flores_plus",
        "split: dev",
        f"revision: {REVISION}",
        *(f"{language}: 3 rows, 1 shard" for language in NINE),
        f"manifest: {out_dir / MANIFEST_FILE}",
    ]
    assert result.stderr == ""


def test_the_revision_of_the_configuration_is_used_when_no_option_says_otherwise(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    config = write_config(tmp_path / "c.yaml", revision="b" * 40)

    result = invoke(fetch_args(tmp_path / "out", "--config", str(config)))

    assert result.exit_code == 0, result.output
    assert {call.extra["revision"] for call in fake.calls} == {"b" * 40}
    assert read_manifest(tmp_path / "out")["status"]["revision"] == "b" * 40


def test_the_option_wins_over_the_revision_of_the_configuration(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    config = write_config(tmp_path / "c.yaml", revision="b" * 40)

    result = invoke(
        fetch_args(tmp_path / "out", "--config", str(config), "--revision", REVISION)
    )

    assert result.exit_code == 0, result.output
    assert {call.extra["revision"] for call in fake.calls} == {REVISION}


def test_the_languages_fetched_are_the_ones_of_the_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = install_fake_load_dataset(monkeypatch, FakeLoadDataset(fixture_rows()))
    config = write_config(tmp_path / "c.yaml")

    result = invoke(
        fetch_args(tmp_path / "out", "--revision", REVISION, "--config", str(config))
    )

    assert result.exit_code == 0, result.output
    assert [call.config for call in fake.calls] == list(LANGUAGES)


# ---- an unpinned fetch --------------------------------------------------------------------


@pytest.mark.parametrize("config_revision", [None, "b" * 40], ids=["none", "pinned"])
def test_the_unpinned_option_fetches_without_a_revision_and_says_so(
    fake: FakeLoadDataset, tmp_path: Path, config_revision: str | None
) -> None:
    config = write_config(tmp_path / "c.yaml", languages=NINE, revision=config_revision)

    result = invoke(fetch_args(tmp_path / "out", "--unpinned", "--config", str(config)))

    assert result.exit_code == 0, result.output
    assert [call.as_tuple() for call in fake.calls] == [
        (DATASET, language, "dev", True, {}) for language in NINE
    ]
    assert read_manifest(tmp_path / "out")["status"]["revision"] == "unpinned"
    assert "revision: unpinned" in result.stdout


def test_a_fetch_without_a_revision_and_without_the_unpinned_option_is_refused_before_any_call(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    result = invoke(fetch_args(tmp_path / "out"))

    assert_one_error(result, 2, "")
    assert "--revision" in result.stderr
    assert "--unpinned" in result.stderr
    assert fake.calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "revision", ["a" * 39, "a" * 41, "A" * 40, "g" * 40, "", "main"]
)
def test_a_revision_that_is_not_forty_lowercase_hexadecimal_characters_is_refused(
    fake: FakeLoadDataset, tmp_path: Path, revision: str
) -> None:
    result = invoke(fetch_args(tmp_path / "out", "--revision", revision))

    assert_one_error(result, 2, "--revision")
    assert fake.calls == []
    assert not (tmp_path / "out").exists()


def test_a_revision_and_the_unpinned_option_together_are_refused(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    result = invoke(fetch_args(tmp_path / "out", "--revision", REVISION, "--unpinned"))

    assert_one_error(result, 2, "")
    assert fake.calls == []


# ---- what the command refuses -------------------------------------------------------------


@pytest.mark.parametrize(
    ("benchmark", "split", "start"),
    [
        ("nothing", "dev", "the benchmark"),
        ("ntrex", "devtest", "the benchmark ntrex"),
        ("flores_plus", "test", "the split"),
        ("human", "dev", "the benchmark"),
    ],
)
def test_a_benchmark_that_cannot_be_fetched_or_a_split_it_does_not_have_is_refused(
    fake: FakeLoadDataset, tmp_path: Path, benchmark: str, split: str, start: str
) -> None:
    args = fetch_args(tmp_path / "out", "--revision", REVISION, split=split)
    args[args.index("--benchmark") + 1] = benchmark

    result = invoke(args)

    assert_one_error(result, 2, start)
    assert fake.calls == []
    assert not (tmp_path / "out").exists()


def test_a_benchmark_with_no_hub_dataset_cannot_be_fetched(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    config = tmp_path / "c.yaml"
    config.write_text(
        "benchmarks:\n"
        "  local_only:\n"
        "    reader: parquet_shards\n"
        "    splits: [dev]\n"
        "    columns: {id: id, text: text}\n"
        "    languages: [khk_Cyrl, eng_Latn]\n",
        encoding="utf-8",
    )
    args = fetch_args(tmp_path / "out", "--revision", REVISION, "--config", str(config))
    args[args.index("--benchmark") + 1] = "local_only"

    result = invoke(args)

    assert_one_error(result, 2, "the benchmark local_only has no Hub dataset")
    assert fake.calls == []


def test_a_missing_configuration_is_refused(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    result = invoke(
        fetch_args(
            tmp_path / "out", "--revision", REVISION, "--config", str(tmp_path / "x")
        )
    )

    assert_one_error(result, 2, "eval suites config not found")


def test_an_output_directory_that_is_the_directory_of_the_configuration_is_refused(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    config = write_config(tmp_path / "c.yaml")

    result = invoke(
        fetch_args(tmp_path, "--revision", REVISION, "--config", str(config))
    )

    assert_one_error(result, 2, "--out-dir")
    assert fake.calls == []


def test_the_manifest_of_another_stage_in_the_output_directory_is_refused_and_left_alone(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    write_foreign_manifest(out_dir)
    before = snapshot(out_dir)

    result = invoke(fetch_args(out_dir, "--revision", REVISION))

    assert_one_error(result, 3, "")
    assert fake.calls == []
    assert snapshot(out_dir) == before


# ---- when the upstream fails --------------------------------------------------------------


def test_an_upstream_error_is_exit_4_naming_the_language_and_the_class_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "hf_" + "S" * 30
    install_fake_load_dataset(
        monkeypatch,
        FakeLoadDataset(
            three_rows(NINE),
            error=ConnectionError(f"cannot reach the Hub with {secret}"),
            error_for="rus_Cyrl",
        ),
    )

    result = invoke(fetch_args(tmp_path / "out", "--revision", REVISION))

    assert_one_error(result, 4, "")
    assert "rus_Cyrl" in result.stderr
    assert "ConnectionError" in result.stderr
    assert secret not in result.output
    assert not (tmp_path / "out" / MANIFEST_FILE).exists()


def test_a_token_in_the_environment_is_never_printed_or_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "hf_" + "T" * 30
    install_fake_load_dataset(
        monkeypatch,
        FakeLoadDataset(
            three_rows(NINE), error=RuntimeError(secret), error_for="eng_Latn"
        ),
    )
    home = tmp_path / "hf_home"
    home.mkdir()
    (home / "token").write_text(secret, encoding="utf-8")

    result = invoke(
        fetch_args(tmp_path / "out", "--revision", REVISION),
        env={"HF_TOKEN": secret, "HF_HOME": str(home)},
    )

    assert result.exit_code == 4
    assert secret not in result.output
    written = [path for path in (tmp_path / "out").rglob("*") if path.is_file()]
    assert all(secret.encode() not in path.read_bytes() for path in written)


def test_a_language_the_hub_returns_no_row_for_is_an_upstream_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = three_rows(NINE)
    rows["kor_Hang"] = []
    install_fake_load_dataset(monkeypatch, FakeLoadDataset(rows))

    result = invoke(fetch_args(tmp_path / "out", "--revision", REVISION))

    assert_one_error(result, 4, "")
    assert "kor_Hang" in result.stderr
    assert not (tmp_path / "out" / MANIFEST_FILE).exists()


def test_a_failed_fetch_leaves_no_manifest_that_vouches_for_the_old_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    install_fake_load_dataset(monkeypatch, FakeLoadDataset(three_rows(NINE)))
    assert invoke(fetch_args(out_dir, "--revision", REVISION)).exit_code == 0
    install_fake_load_dataset(
        monkeypatch,
        FakeLoadDataset(three_rows(NINE), error=OSError("down"), error_for="deu_Latn"),
    )

    result = invoke(fetch_args(out_dir, "--revision", REVISION))

    assert result.exit_code == 4
    assert not (out_dir / MANIFEST_FILE).exists()


def test_a_manifest_that_cannot_be_written_is_exit_1(
    fake: FakeLoadDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_replace(monkeypatch, MANIFEST_FILE)

    result = invoke(fetch_args(tmp_path / "out", "--revision", REVISION))

    assert_one_error(result, 1, "cannot write the")
    assert not (tmp_path / "out" / MANIFEST_FILE).exists()


def test_an_output_directory_that_cannot_be_made_is_exit_1_before_any_call(
    fake: FakeLoadDataset, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file", encoding="utf-8")

    result = invoke(fetch_args(blocker / "out", "--revision", REVISION))

    assert_one_error(result, 1, "cannot write the")
    assert fake.calls == []


# ---- a second fetch -----------------------------------------------------------------------


def test_a_second_fetch_removes_the_shards_the_old_manifest_lists_and_no_other_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    install_fake_load_dataset(monkeypatch, FakeLoadDataset(three_rows(NINE)))
    assert invoke(fetch_args(out_dir, "--revision", REVISION)).exit_code == 0
    stray = out_dir / "dev" / "eng_Latn" / "notes.txt"
    stray.write_text("kept", encoding="utf-8")
    other_split = out_dir / "devtest" / "eng_Latn"
    other_split.mkdir(parents=True)
    unlisted = other_split / SHARD
    unlisted.write_bytes(b"kept too")
    install_fake_load_dataset(monkeypatch, FakeLoadDataset(fixture_rows()))
    config = write_config(tmp_path / "three.yaml")

    result = invoke(
        fetch_args(out_dir, "--revision", REVISION, "--config", str(config))
    )

    assert result.exit_code == 0, result.output
    assert not (out_dir / "dev" / "zho_Hans" / SHARD).exists()
    assert pl.read_parquet(out_dir / "dev" / "eng_Latn" / SHARD).height == 4
    assert stray.read_text(encoding="utf-8") == "kept"
    assert unlisted.read_bytes() == b"kept too"
    assert [item["path"] for item in read_manifest(out_dir)["outputs"]] == [
        str(out_dir / "dev" / language / SHARD) for language in LANGUAGES
    ]
