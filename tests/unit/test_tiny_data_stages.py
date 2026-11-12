"""`rinchen.tiny.data_stages`: the environment, fetch and corpus stages of the tiny pipeline (D15,
spec P2, P4, P15, brief G3.4, G3.5; the filter stage is in `test_tiny_filter_stage.py`).

The stages run in process and start no command (the runner of the context fails the test if it is
called) and no network (the `no_network` guard). Each one writes only into its own directory and
takes everything it reads from the context: the profile, the parsed fixtures and the output
directory. The tests run from a temporary working directory, so a stage that read a relative path
of the repository would not find it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from rinchen.acquire.registry import LocalSourceEntry, fetch_source, load_registry
from rinchen.config.tiny import FixturesConfig, TinyConfig
from rinchen.corpus.readers import scan_corpus
from rinchen.errors import ConfigError, DataError
from rinchen.maintain.lock import read_lock
from rinchen.tiny import data_stages
from rinchen.tiny.context import StageContext, StageResult
from rinchen.tiny.corpus import pair_rows
from rinchen.tiny.data_stages import (
    run_corpus,
    run_environment,
    run_fetch,
)
from rinchen.tiny.fixtures import FIXTURE_FILES, Fixtures
from rinchen.tiny.states import FAIL_INPUT, fail_environment, is_closed_reason
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_stage_inputs import (
    LOCK_PATH,
    make_context,
    shipped_config,
    shipped_fixtures,
)

pytestmark = pytest.mark.usefixtures("no_network")

CREATED: Final = datetime(2026, 11, 11, tzinfo=UTC)
FIXTURE_BYTES: Final = 67_104


@pytest.fixture(autouse=True)
def work_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run from a directory that holds nothing: a stage must not read the repository by itself."""
    here = tmp_path / "cwd"
    here.mkdir()
    monkeypatch.chdir(here)
    yield


def names_in(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def changed_config(**sections: Any) -> TinyConfig:
    return shipped_config().model_copy(update=sections)


def fixture_texts(fixtures: Fixtures) -> set[str]:
    texts = {r.src for rows in fixtures.pairs.values() for r in rows}
    texts |= {r.tgt for rows in fixtures.pairs.values() for r in rows}
    texts |= {r.mong for r in fixtures.mong} | {r.eng for r in fixtures.mong}
    texts |= {r.khk for r in fixtures.mong} | {r.eng for r in fixtures.mono}
    texts |= {r.src for r in fixtures.noise} | {r.tgt for r in fixtures.noise}
    return {text for text in texts if len(text) > 4}


# ---- G3.4: the environment stage ---------------------------------------------------------------


def lock_bytes() -> bytes:
    return LOCK_PATH.read_bytes()


def locked_table() -> dict[str, tuple[str, ...]]:
    return {
        name: versions[:1]
        for name, versions in read_lock(lock_bytes()).versions.items()
    }


def test_the_real_lock_and_the_real_environment_are_ok_with_186_installed(
    tmp_path: Path,
) -> None:
    result = run_environment(make_context(tmp_path / "out"), lock_bytes())

    assert result.record.id == "environment"
    assert result.record.state == "ok"
    assert result.record.reason is None
    assert result.record.detail == result.facts
    assert result.facts == {
        "installed": 186,
        "locked": 186,
        "mismatched": 0,
        "extra": 0,
        "not_installed": 0,
    }


def test_the_stage_writes_nothing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    run_environment(make_context(out_dir), lock_bytes())

    assert not out_dir.exists()


def test_an_environment_that_is_the_lock_is_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = locked_table()
    monkeypatch.setattr(data_stages, "installed_versions", lambda: table)

    result = run_environment(make_context(tmp_path), lock_bytes())

    assert result.record.state == "ok"
    assert result.facts["installed"] == len(table)


def test_one_package_at_another_version_fails_and_the_message_holds_counts_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = locked_table()
    name = sorted(table)[10]
    version = table[name][0]
    table[name] = ("0.0.0.dev9",)
    monkeypatch.setattr(data_stages, "installed_versions", lambda: table)

    result = run_environment(make_context(tmp_path), lock_bytes())

    assert result.record.state == "fail"
    assert result.record.reason == fail_environment(1, 0)
    assert is_closed_reason(result.record.reason or "")
    assert result.facts["mismatched"] == 1
    shown = repr(result)
    assert name not in shown
    assert version not in shown
    assert "0.0.0.dev9" not in shown


def test_a_package_the_lock_does_not_hold_fails_as_an_extra_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = {**locked_table(), "zz-planted-package": ("1.0",), "zz-other": ("2",)}
    monkeypatch.setattr(data_stages, "installed_versions", lambda: table)

    result = run_environment(make_context(tmp_path), lock_bytes())

    assert result.record.state == "fail"
    assert result.record.reason == fail_environment(0, 2)
    assert "zz-planted-package" not in repr(result)
    assert result.facts["extra"] == 2


def test_a_package_that_is_locked_and_not_installed_is_only_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = locked_table()
    for name in sorted(table)[:3]:
        del table[name]
    monkeypatch.setattr(data_stages, "installed_versions", lambda: table)

    result = run_environment(make_context(tmp_path), lock_bytes())

    assert result.record.state == "ok"
    assert result.facts["not_installed"] == 3


def test_bytes_that_are_not_a_lock_are_a_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="uv.lock"):
        run_environment(make_context(tmp_path), b"not = [a lock")


# ---- G3.5: the fetch stage ---------------------------------------------------------------------


def test_the_seven_fixtures_are_copied_byte_for_byte_into_the_stage_directory(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    config = shipped_config()

    result = run_fetch(make_context(out_dir))

    assert result.record.id == "fetch"
    assert result.record.state == "ok"
    assert result.record.detail == {"files": 7, "bytes": FIXTURE_BYTES}
    assert result.facts == result.record.detail
    copies = out_dir / "01_fetch" / "files"
    assert names_in(copies) == sorted(FIXTURE_FILES)
    for name in FIXTURE_FILES:
        original = (config.fixtures.directory / name).read_bytes()
        assert (copies / name).read_bytes() == original
        assert hashlib.sha256(original).hexdigest() == config.fixtures.sha256[name]


def test_the_stage_writes_only_into_its_own_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    run_fetch(make_context(out_dir))

    assert names_in(out_dir) == ["01_fetch"]
    assert names_in(out_dir / "01_fetch") == ["files"]


def with_changed_pin(name: str) -> TinyConfig:
    config = shipped_config()
    digest = config.fixtures.sha256[name]
    pins = {
        **config.fixtures.sha256,
        name: ("0" if digest[0] != "0" else "1") + digest[1:],
    }
    return changed_config(
        fixtures=FixturesConfig(directory=config.fixtures.directory, sha256=pins)
    )


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_a_changed_pin_is_a_data_error_before_any_copy(
    tmp_path: Path, name: str
) -> None:
    out_dir = tmp_path / "out"

    with pytest.raises(DataError, match="pin"):
        run_fetch(make_context(out_dir, config=with_changed_pin(name)))

    assert not out_dir.exists()


def write_registry(path: Path, urls: list[str]) -> Path:
    entries = [
        {
            "id": f"entry-{index}",
            "kind": "local",
            "langs": ["khk_Cyrl"],
            "url": url,
        }
        for index, url in enumerate(urls)
    ]
    path.write_text(yaml.safe_dump({"sources": entries}), encoding="utf-8")
    return path


def fixture_urls(prefix: str = "local://tests/fixtures/parallel/") -> list[str]:
    return [prefix + name for name in FIXTURE_FILES]


def test_an_entry_outside_the_fixtures_directory_fails_and_nothing_is_copied(
    tmp_path: Path,
) -> None:
    urls = fixture_urls()
    urls[2] = "local://tests/fixtures/script_conv/tiny_words.tsv"
    registry = write_registry(tmp_path / "registry.yaml", urls)
    out_dir = tmp_path / "out"

    result = run_fetch(make_context(out_dir, config=changed_config(registry=registry)))

    assert result.record.state == "fail"
    assert result.record.reason == FAIL_INPUT
    assert not (out_dir / "01_fetch").exists() or not list(
        (out_dir / "01_fetch").rglob("*.tsv")
    )


def test_a_registry_of_the_right_names_in_another_directory_fails(
    tmp_path: Path,
) -> None:
    config = shipped_config()
    copied = tmp_path / "copy"
    copied.mkdir()
    for name in FIXTURE_FILES:
        (copied / name).write_bytes((config.fixtures.directory / name).read_bytes())
    registry = write_registry(
        tmp_path / "registry.yaml", fixture_urls(f"local://{copied}/")
    )
    out_dir = tmp_path / "out"

    result = run_fetch(make_context(out_dir, config=changed_config(registry=registry)))

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert not out_dir.exists()


def fixtures_directory_with_alias(tmp_path: Path) -> tuple[Path, TinyConfig]:
    """A copy of the fixtures directory that also holds `alias.tsv`, a link to `noise.tsv`, and the
    configuration that names it; the registry is left for the test to write."""
    config = shipped_config()
    copied = tmp_path / "fixtures"
    copied.mkdir()
    for name in FIXTURE_FILES:
        (copied / name).write_bytes((config.fixtures.directory / name).read_bytes())
    (copied / "alias.tsv").symlink_to("noise.tsv")
    moved = changed_config(
        fixtures=FixturesConfig(directory=copied, sha256=config.fixtures.sha256)
    )
    return copied, moved


def test_an_entry_that_is_a_link_to_a_fixture_under_another_name_fails_and_nothing_is_copied(
    tmp_path: Path,
) -> None:
    copied, config = fixtures_directory_with_alias(tmp_path)
    urls = fixture_urls(f"local://{copied}/")
    urls[FIXTURE_FILES.index("noise.tsv")] = f"local://{copied}/alias.tsv"
    registry = write_registry(tmp_path / "registry.yaml", urls)
    out_dir = tmp_path / "out"

    result = run_fetch(
        make_context(out_dir, config=config.model_copy(update={"registry": registry}))
    )

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert result.facts == {"files": 0, "bytes": 0}
    assert not out_dir.exists()


def test_a_fixture_that_is_itself_a_link_to_a_file_of_another_name_fails_and_nothing_is_copied(
    tmp_path: Path,
) -> None:
    copied, config = fixtures_directory_with_alias(tmp_path)
    (copied / "noise.tsv").rename(copied / "noise_original.tsv")
    (copied / "noise.tsv").symlink_to("noise_original.tsv")
    registry = write_registry(
        tmp_path / "registry.yaml", fixture_urls(f"local://{copied}/")
    )
    out_dir = tmp_path / "out"

    result = run_fetch(
        make_context(out_dir, config=config.model_copy(update={"registry": registry}))
    )

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert not out_dir.exists()


def test_the_name_the_check_expects_for_a_copy_is_the_name_fetch_source_gives_it(
    tmp_path: Path,
) -> None:
    for entry in load_registry(shipped_config().registry):
        assert isinstance(entry, LocalSourceEntry)
        copy = fetch_source(entry, tmp_path / entry.id)
        assert data_stages._copy_name(entry) == copy.name


def test_an_address_with_a_slash_at_its_end_is_named_by_its_last_segment_as_fetch_source_does(
    tmp_path: Path,
) -> None:
    urls = [url + "/" for url in fixture_urls()]
    registry = write_registry(tmp_path / "registry.yaml", urls)

    result = run_fetch(
        make_context(tmp_path / "out", config=changed_config(registry=registry))
    )

    assert result.record.state == "ok"
    assert names_in(tmp_path / "out" / "01_fetch" / "files") == sorted(FIXTURE_FILES)


def test_the_same_registry_with_the_name_of_the_fixture_is_fetched(
    tmp_path: Path,
) -> None:
    copied, config = fixtures_directory_with_alias(tmp_path)
    registry = write_registry(
        tmp_path / "registry.yaml", fixture_urls(f"local://{copied}/")
    )

    result = run_fetch(
        make_context(
            tmp_path / "out", config=config.model_copy(update={"registry": registry})
        )
    )

    assert result.record.state == "ok"
    assert names_in(tmp_path / "out" / "01_fetch" / "files") == sorted(FIXTURE_FILES)


def test_an_entry_that_climbs_out_of_the_directory_fails(tmp_path: Path) -> None:
    urls = fixture_urls()
    urls[0] = "local://tests/fixtures/parallel/../script_conv/tiny_words.tsv"
    registry = write_registry(tmp_path / "registry.yaml", urls)

    result = run_fetch(
        make_context(tmp_path / "out", config=changed_config(registry=registry))
    )

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)


def test_an_entry_of_another_kind_fails_and_opens_no_connection(tmp_path: Path) -> None:
    # Cut after the eight characters of a scheme, the address of the remote entry reads as the path
    # of a fixture of the directory: only its kind and its scheme tell it is not a local file.
    registry = tmp_path / "registry.yaml"
    entries = [
        {"id": f"e{i}", "kind": "local", "langs": ["khk_Cyrl"], "url": url}
        for i, url in enumerate(fixture_urls()[:-1])
    ]
    entries.append(
        {
            "id": "remote",
            "kind": "http",
            "langs": ["khk_Cyrl"],
            "url": "https://tests/fixtures/parallel/entities.tsv",
        }
    )
    registry.write_text(yaml.safe_dump({"sources": entries}), encoding="utf-8")

    result = run_fetch(
        make_context(tmp_path / "out", config=changed_config(registry=registry))
    )

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert not (tmp_path / "out" / "01_fetch").exists()


@pytest.mark.parametrize("drop", range(len(FIXTURE_FILES)))
def test_a_registry_that_misses_a_fixture_fails(tmp_path: Path, drop: int) -> None:
    urls = fixture_urls()
    del urls[drop]
    registry = write_registry(tmp_path / "registry.yaml", urls)

    result = run_fetch(
        make_context(tmp_path / "out", config=changed_config(registry=registry))
    )

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)


def test_a_registry_that_lists_a_fixture_twice_fails(tmp_path: Path) -> None:
    registry = write_registry(
        tmp_path / "registry.yaml", [*fixture_urls(), fixture_urls()[0]]
    )

    result = run_fetch(
        make_context(tmp_path / "out", config=changed_config(registry=registry))
    )

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)


def test_an_entry_for_a_file_that_is_no_fixture_fails(tmp_path: Path) -> None:
    urls = fixture_urls()
    urls[1] = "local://tests/fixtures/parallel/extra.tsv"
    registry = write_registry(tmp_path / "registry.yaml", urls)

    result = run_fetch(
        make_context(tmp_path / "out", config=changed_config(registry=registry))
    )

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)


def test_a_registry_that_cannot_be_read_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        run_fetch(
            make_context(
                tmp_path / "out",
                config=changed_config(registry=tmp_path / "absent.yaml"),
            )
        )


def test_a_copy_that_differs_from_its_pin_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fetch = data_stages.fetch_source

    def damaged(entry: Any, dest_dir: Path) -> Path:
        copy = real_fetch(entry, dest_dir)
        if copy.name == "noise.tsv":
            copy.write_bytes(copy.read_bytes() + b"x")
        return copy

    monkeypatch.setattr(data_stages, "fetch_source", damaged)

    result = run_fetch(make_context(tmp_path / "out"))

    assert (result.record.state, result.record.reason) == ("fail", FAIL_INPUT)
    assert result.record.detail["files"] == 4


def test_the_stage_follows_the_context_and_not_the_repository(tmp_path: Path) -> None:
    config = shipped_config()
    copied = tmp_path / "elsewhere"
    copied.mkdir()
    for name in FIXTURE_FILES:
        (copied / name).write_bytes((config.fixtures.directory / name).read_bytes())
    registry = write_registry(
        tmp_path / "registry.yaml", fixture_urls(f"local://{copied}/")
    )
    moved = changed_config(
        registry=registry,
        fixtures=FixturesConfig(directory=copied, sha256=config.fixtures.sha256),
    )

    result = run_fetch(make_context(tmp_path / "out", config=moved))

    assert result.record.state == "ok"
    assert names_in(tmp_path / "out" / "01_fetch" / "files") == sorted(FIXTURE_FILES)


# ---- the corpus stage --------------------------------------------------------------------------


def run_corpus_in(out_dir: Path, **context: Any) -> tuple[StageContext, StageResult]:
    ctx = make_context(out_dir, **context)
    return ctx, run_corpus(ctx)


def test_the_corpus_stage_writes_the_pair_corpus_and_its_counts(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    ctx, result = run_corpus_in(out_dir)

    assert result.record.id == "corpus"
    assert result.record.state == "ok"
    assert result.record.detail == {"rows": 770, "dropped": 0}
    assert names_in(out_dir) == ["02_corpus"]
    assert names_in(out_dir / "02_corpus") == ["corpus.parquet", "counts.json"]
    written = scan_corpus(out_dir / "02_corpus" / "corpus.parquet").collect()
    assert written.equals(pair_rows(ctx.fixtures, CREATED))
    assert written["created"].unique().to_list() == [CREATED]


def test_the_counts_are_the_facts_and_name_no_text(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    _, result = run_corpus_in(out_dir)

    raw = (out_dir / "02_corpus" / "counts.json").read_bytes()
    assert raw.isascii()
    assert raw.endswith(b"\n")
    counts = json.loads(raw)
    assert counts == json.loads(json.dumps(result.facts))
    assert raw == (json.dumps(counts, sort_keys=True, indent=2) + "\n").encode()
    assert counts["rows"] == 770
    assert counts["dropped"] == 0
    assert counts["dropped_by_corpus"] == {
        "tiny-eng_Latn-khk_Cyrl": 0,
        "tiny-mn_Mong": 0,
        "tiny-noise": 0,
        "tiny-rus_Cyrl-khk_Cyrl": 0,
    }
    assert counts["splits"] == {"dev": 40, "holdout": 10, "test": 70, "train": 650}
    assert counts["corpora"] == {
        "tiny-eng_Latn-khk_Cyrl": 345,
        "tiny-mn_Mong": 70,
        "tiny-noise": 10,
        "tiny-rus_Cyrl-khk_Cyrl": 345,
    }
    text = raw.decode()
    assert not any(sentence in text for sentence in fixture_texts(shipped_fixtures()))
    assert not any(
        sentence in repr(result) for sentence in fixture_texts(shipped_fixtures())
    )


def test_the_stamp_is_the_one_of_the_profile(tmp_path: Path) -> None:
    other = changed_config(created="2031-02-03T04:05:06Z")

    _, result = run_corpus_in(tmp_path / "out", config=other)

    written = scan_corpus(tmp_path / "out" / "02_corpus" / "corpus.parquet").collect()
    assert written["created"].unique().to_list() == [
        datetime(2031, 2, 3, 4, 5, 6, tzinfo=UTC)
    ]
    assert result.record.state == "ok"


def test_the_corpus_follows_the_fixtures_of_the_context(tmp_path: Path) -> None:
    fixtures = shipped_fixtures()
    pairs = {
        **fixtures.pairs,
        "eng_Latn-khk_Cyrl": fixtures.pairs["eng_Latn-khk_Cyrl"][:10],
    }
    fewer = dataclasses.replace(fixtures, pairs=pairs)

    _, result = run_corpus_in(tmp_path / "out", fixtures=fewer)

    assert result.record.detail == {"rows": 435, "dropped": 0}


def test_a_pair_that_is_not_one_sentence_is_dropped_and_counted_and_the_stage_is_ok(
    tmp_path: Path,
) -> None:
    fixtures = shipped_fixtures()
    key = "rus_Cyrl-khk_Cyrl"
    rows = list(fixtures.pairs[key])
    rows[5] = dataclasses.replace(rows[5], src="   ")
    changed = dataclasses.replace(fixtures, pairs={**fixtures.pairs, key: tuple(rows)})

    _, result = run_corpus_in(tmp_path / "out", fixtures=changed)

    assert result.record.state == "ok"
    assert result.record.detail == {"rows": 769, "dropped": 1}
    counts = json.loads((tmp_path / "out" / "02_corpus" / "counts.json").read_bytes())
    assert counts["dropped_by_corpus"]["tiny-rus_Cyrl-khk_Cyrl"] == 1


def test_the_corpus_file_is_equal_whatever_the_directory(tmp_path: Path) -> None:
    run_corpus_in(tmp_path / "first")
    run_corpus_in(tmp_path / "second")

    for name in ("corpus.parquet", "counts.json"):
        assert (tmp_path / "first" / "02_corpus" / name).read_bytes() == (
            tmp_path / "second" / "02_corpus" / name
        ).read_bytes()
