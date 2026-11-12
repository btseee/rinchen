"""The inputs of `rinchen tiny run` (D15 Task 7, spec P11 and P13): the profile's pipeline file, the seven
fixtures, the Eole base config, the serve config, the converter's words file, the profile's own file (it
holds the seed) and `uv.lock`, each hashed BEFORE it is read, and the config, the lock, the seed and the
fixtures parsed from the bytes that were hashed, so that a file that changed in between is refused and
never read.

The pipeline file is the one file read before the rest is hashed, because it says where the rest are; it is
hashed first and read through `verified_content`. A fixture whose digest differs from its pin, or is
missing, is a `DataError` (exit 3); a file that changed after it was hashed is a `ConfigError` (exit 2).
The tests that change a file between the hashing and the reading wrap the hashing function of the module
and rewrite a copy in a temporary directory right after it has hashed it.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from rinchen.cli.tiny_inputs import (
    HashedInputs,
    hash_inputs,
    pipeline_config_path,
    read_inputs,
)
from rinchen.config.loader import REPO_ROOT
from rinchen.errors import ConfigError, DataError
from rinchen.tiny.fixtures import FIXTURE_FILES
from tests.support.no_network import no_network  # noqa: F401 - the fixture of the mark
from tests.support.tiny_cli import append_a_line, changing_after_hashing, make_workspace

pytestmark = pytest.mark.usefixtures("no_network")

SHIPPED = REPO_ROOT / "configs" / "tiny" / "tiny.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Copies of every file of the inputs a test may change, by name (see `make_workspace`)."""
    return dataclasses.asdict(make_workspace(tmp_path, monkeypatch))


# ---- which file is the pipeline's ---------------------------------------------------------------------------


def test_the_profile_names_the_file_of_the_repository() -> None:
    assert pipeline_config_path("tiny", None) == SHIPPED


def test_a_config_option_replaces_the_file_of_the_profile(tmp_path: Path) -> None:
    other = tmp_path / "other.yaml"

    assert pipeline_config_path("tiny", other) == other
    # The profile of an explicit file need not have a pipeline file of its own.
    assert pipeline_config_path("full", other) == other


@pytest.mark.parametrize(
    "name",
    ["Full", "FULL", "a/b", "../tiny", "", "1x", "a-b", "x" * 17, "tiny.yaml", "tiny "],
)
def test_a_profile_name_outside_the_pattern_is_a_config_error(name: str) -> None:
    with pytest.raises(ConfigError, match="profile name"):
        pipeline_config_path(name, None)


def test_a_profile_with_no_pipeline_file_is_refused_with_the_message_of_the_spec() -> (
    None
):
    with pytest.raises(ConfigError) as raised:
        pipeline_config_path("full", None)

    message = str(raised.value)
    assert "no pipeline configuration for the profile 'full'" in message
    assert "(configs/tiny/full.yaml)" in message
    assert "docs/runbooks/15_tiny_pipeline.md" in message
    assert str(REPO_ROOT) not in message


def test_the_message_is_built_from_the_profile_and_not_from_a_name_in_the_code() -> (
    None
):
    with pytest.raises(ConfigError) as raised:
        pipeline_config_path("another_one", None)

    assert "'another_one'" in str(raised.value)
    assert "(configs/tiny/another_one.yaml)" in str(raised.value)


# ---- hashing -----------------------------------------------------------------------------------------------


def test_the_shipped_inputs_are_hashed_in_the_order_of_the_manifest() -> None:
    hashed = hash_inputs("tiny", SHIPPED)

    assert isinstance(hashed, HashedInputs)
    expected = [
        SHIPPED,
        *(
            REPO_ROOT / "tests" / "fixtures" / "parallel" / name
            for name in FIXTURE_FILES
        ),
        REPO_ROOT / "configs" / "eole" / "tiny.yaml",
        REPO_ROOT / "configs" / "serve" / "tiny.yaml",
        REPO_ROOT / "tests" / "fixtures" / "script_conv" / "tiny_words.tsv",
        REPO_ROOT / "configs" / "profiles" / "tiny.yaml",
        REPO_ROOT / "uv.lock",
    ]
    entries = hashed.entries()
    assert [entry.path for entry in entries] == [str(path) for path in expected]
    assert [entry.sha256 for entry in entries] == [sha256(path) for path in expected]
    assert [entry.bytes for entry in entries] == [
        path.stat().st_size for path in expected
    ]
    assert len(entries) == 13


def test_the_fixtures_are_listed_by_name_with_the_digests_that_were_hashed() -> None:
    hashed = hash_inputs("tiny", SHIPPED)

    assert list(hashed.fixtures) == list(FIXTURE_FILES)
    assert {
        name: entry.sha256 for name, entry in hashed.fixtures.items()
    } == hashed.config.fixtures.sha256


def test_the_config_is_the_one_parsed_from_the_bytes_that_were_hashed(
    workspace: dict[str, Path],
) -> None:
    hashed = hash_inputs("tiny", workspace["config"])

    assert hashed.config_file.sha256 == sha256(workspace["config"])
    assert hashed.config.fixtures.directory == workspace["fixtures"]
    assert hashed.config.train.base == workspace["base"]


# ---- refusals before anything is read ---------------------------------------------------------------------


def test_a_config_file_that_does_not_exist_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="tiny config"):
        hash_inputs("tiny", tmp_path / "missing.yaml")


def test_a_config_that_is_not_valid_is_a_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("fixtures: 3\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid tiny config"):
        hash_inputs("tiny", bad)


def test_a_fixture_whose_digest_differs_from_its_pin_is_a_data_error_naming_the_file(
    workspace: dict[str, Path],
) -> None:
    path = workspace["fixtures"] / "noise.tsv"
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(DataError) as raised:
        hash_inputs("tiny", workspace["config"])

    assert "noise.tsv" in str(raised.value)
    assert "pin" in str(raised.value)


def test_a_missing_fixture_is_a_data_error(workspace: dict[str, Path]) -> None:
    (workspace["fixtures"] / "nouns.tsv").unlink()

    with pytest.raises(DataError, match="nouns.tsv"):
        hash_inputs("tiny", workspace["config"])


@pytest.mark.parametrize(
    ("key", "error"),
    [
        ("base", ConfigError),
        ("serve", ConfigError),
        ("words", ConfigError),
        ("profile", ConfigError),
        ("lock", DataError),
    ],
)
def test_a_missing_input_is_refused_with_the_error_of_its_kind(
    key: str, error: type[Exception], workspace: dict[str, Path]
) -> None:
    workspace[key].unlink()

    with pytest.raises(error):
        hash_inputs("tiny", workspace["config"])


def test_the_digests_are_taken_before_the_files_of_the_run_are_read(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing but the pipeline file is opened until every file is hashed: the lock and the profile
    file are read only by `read_inputs`."""
    opened: list[str] = []
    real = Path.read_bytes

    def spying(self: Path) -> bytes:
        opened.append(self.name)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", spying)

    hash_inputs("tiny", workspace["config"])

    assert workspace["lock"].name not in opened
    assert workspace["profile"].name not in opened
    assert FIXTURE_FILES[0] not in opened


# ---- reading what was hashed ---------------------------------------------------------------------------------


def test_the_lock_the_seed_and_the_fixtures_are_parsed_from_the_bytes_that_were_hashed(
    workspace: dict[str, Path],
) -> None:
    hashed = hash_inputs("tiny", workspace["config"])

    read = read_inputs(hashed)

    assert read.lock_bytes == workspace["lock"].read_bytes()
    assert read.lock.entries == 187
    assert len(read.lock.versions) == 186
    assert read.seed == 1234
    assert read.config == hashed.config
    assert len(read.fixtures.pairs["eng_Latn-khk_Cyrl"]) == 345


@pytest.mark.parametrize("key", ["lock", "profile", "fixtures"])
def test_a_file_that_changed_after_it_was_hashed_is_a_config_error_and_is_not_read(
    key: str, workspace: dict[str, Path]
) -> None:
    hashed = hash_inputs("tiny", workspace["config"])
    target = (
        workspace["fixtures"] / "noise.tsv" if key == "fixtures" else workspace[key]
    )
    append_a_line(target)

    with pytest.raises(ConfigError, match="changed"):
        read_inputs(hashed)


def test_the_config_that_changed_between_its_hashing_and_its_reading_is_a_config_error(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    changing_after_hashing(monkeypatch, workspace["config"], append_a_line)

    with pytest.raises(ConfigError, match="changed"):
        hash_inputs("tiny", workspace["config"])


def test_a_fixture_that_changed_between_the_hashing_and_the_reading_is_a_config_error_naming_the_file(
    workspace: dict[str, Path],
) -> None:
    hashed = hash_inputs("tiny", workspace["config"])
    path = workspace["fixtures"] / "entities.tsv"
    path.write_bytes(path.read_bytes() + b"y")

    with pytest.raises(ConfigError, match="entities.tsv"):
        read_inputs(hashed)


def test_a_lock_that_is_not_a_lock_is_a_data_error(workspace: dict[str, Path]) -> None:
    workspace["lock"].write_text("this is not toml = = =\n", encoding="utf-8")

    hashed = hash_inputs("tiny", workspace["config"])

    with pytest.raises(DataError, match="uv.lock"):
        read_inputs(hashed)


# ---- the seed ----------------------------------------------------------------------------------------------


def seed_of(workspace: dict[str, Path], text: str) -> int:
    workspace["profile"].write_text(text, encoding="utf-8")
    return read_inputs(hash_inputs("tiny", workspace["config"])).seed


def test_the_seed_is_the_integer_of_the_profile_file_and_nothing_of_the_environment(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RINCHEN_SEED", "99")

    assert seed_of(workspace, "seed: 7\npaths:\n  data_root: data/tiny\n") == 7
    assert seed_of(workspace, "seed: 0\n") == 0


@pytest.mark.parametrize(
    "text",
    [
        "paths: {}\n",
        "seed: true\n",
        "seed: false\n",
        "seed: '1234'\n",
        "seed: 1.5\n",
        "seed: -1\n",
        "seed: null\n",
        "- seed\n",
        "seed: [1]\n",
        "seed: {a: 1\n",
        "",
        "42\n",
    ],
    ids=[
        "missing",
        "true",
        "false",
        "text",
        "float",
        "negative",
        "null",
        "list",
        "sequence",
        "yaml",
        "empty",
        "scalar",
    ],
)
def test_a_seed_that_is_not_a_non_negative_integer_is_a_config_error(
    text: str, workspace: dict[str, Path]
) -> None:
    workspace["profile"].write_text(text, encoding="utf-8")
    hashed = hash_inputs("tiny", workspace["config"])

    with pytest.raises(ConfigError, match="seed"):
        read_inputs(hashed)


def test_the_message_about_a_seed_quotes_nothing_of_the_file(
    workspace: dict[str, Path],
) -> None:
    workspace["profile"].write_text("seed: not-a-number-xyz\n", encoding="utf-8")
    hashed = hash_inputs("tiny", workspace["config"])

    with pytest.raises(ConfigError) as raised:
        read_inputs(hashed)

    assert "not-a-number-xyz" not in str(raised.value)


def test_the_profile_file_of_another_profile_is_the_one_whose_seed_is_read(
    workspace: dict[str, Path],
) -> None:
    other = workspace["profile"].with_name("other.yaml")
    other.write_text("seed: 5\n", encoding="utf-8")

    assert read_inputs(hash_inputs("other", workspace["config"])).seed == 5
