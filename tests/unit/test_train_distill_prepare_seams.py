"""The two seams `rinchen distill prepare` needs of the library: the digests of the KD files a
caller took itself (`load_kd_rows(..., digests=)`, `prepare_student(..., kd_digests=)`), so
that a command that vouches for its inputs hashes a large KD file once, and the hook that runs
after every refusal and before the first write (`prepare_student(..., before_writing=)`), where
the command removes the old manifest and makes the output directory.

The behaviour without them is in `test_train_distill_corpus.py` and
`test_train_distill_prepare*.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import rinchen.train.distill_corpus as distill_corpus_module
import rinchen.train.distill_prepare as distill_prepare_module
from rinchen.errors import ConfigError
from rinchen.manifest.stage import sha256_of
from rinchen.train.distill_corpus import KD_EXCLUSION_REASONS, load_kd_rows
from tests.support.corpus_rows import EN_MN, MN_EN, MN_RU, RU_MN
from tests.support.planted_student import (
    Inputs,
    build_inputs,
    files_under,
    prepare,
)
from tests.support.step_dirs import flip_one_byte

DIRECTIONS = (EN_MN, MN_EN, MN_RU, RU_MN)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    return build_inputs(tmp_path_factory.mktemp("distill_prepare_seams_inputs"))


class HashSpy:
    """Counts the calls of `distill_corpus.sha256_of`, the hashing of a KD file, by path."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.paths: list[Path] = []
        real = distill_corpus_module.sha256_of

        def counting(path: Path) -> str:
            self.paths.append(path)
            return real(path)

        monkeypatch.setattr(distill_corpus_module, "sha256_of", counting)


# --- the digests the caller took ---------------------------------------------------------------


def test_without_digests_each_kd_file_is_hashed_once(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = HashSpy(monkeypatch)

    load_kd_rows([inputs.kd], directions=DIRECTIONS)

    assert spy.paths == [inputs.kd]


def test_given_digests_that_match_the_manifest_the_file_is_not_hashed_again(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = HashSpy(monkeypatch)

    loaded = load_kd_rows(
        [inputs.kd], directions=DIRECTIONS, digests={inputs.kd: sha256_of(inputs.kd)}
    )

    assert spy.paths == []
    assert loaded.stats.rows_kept == 3600
    assert loaded.stats.excluded == dict.fromkeys(KD_EXCLUSION_REASONS, 0)


def test_a_given_digest_that_is_not_the_one_the_manifest_vouches_for_is_a_config_error(
    inputs: Inputs,
) -> None:
    wrong = "0" * 64

    with pytest.raises(ConfigError) as caught:
        load_kd_rows([inputs.kd], directions=DIRECTIONS, digests={inputs.kd: wrong})

    message = str(caught.value)
    assert str(inputs.kd) in message
    assert wrong in message
    assert sha256_of(inputs.kd) in message
    assert "the manifest vouches for" in message


def test_a_digest_of_bytes_that_changed_after_the_generation_wrote_them_is_refused(
    inputs: Inputs, tmp_path: Path
) -> None:
    kd = tmp_path / "kd.parquet"
    kd.write_bytes(inputs.kd.read_bytes())
    (tmp_path / "manifest.json").write_bytes(
        (inputs.kd.parent / "manifest.json").read_bytes()
    )
    flip_one_byte(kd)

    with pytest.raises(ConfigError, match="the manifest vouches for"):
        load_kd_rows([kd], directions=DIRECTIONS, digests={kd: sha256_of(kd)})


def test_digests_must_cover_every_file(inputs: Inputs, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    second = other / inputs.kd.name
    second.write_bytes(inputs.kd.read_bytes())

    with pytest.raises(ValueError, match=r"no digest.*other"):
        load_kd_rows(
            [inputs.kd, second],
            directions=DIRECTIONS,
            digests={inputs.kd: sha256_of(inputs.kd)},
        )


def test_prepare_student_hands_the_digests_to_the_loader(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = HashSpy(monkeypatch)

    prepare(
        inputs,
        tmp_path / "out",
        kd_digests={inputs.kd: sha256_of(inputs.kd)},
    )

    assert spy.paths == []
    assert (tmp_path / "out" / "config.yaml").is_file()


def test_prepare_student_refuses_a_digest_the_manifest_does_not_vouch_for_and_writes_nothing(
    inputs: Inputs, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="the manifest vouches for"):
        prepare(inputs, tmp_path / "out", kd_digests={inputs.kd: "f" * 64})

    assert not (tmp_path / "out").exists()


# --- the hook before the first write -----------------------------------------------------------


class Hook:
    """A `before_writing` hook that records what it saw: whether the output directory existed
    and how many KD files had been loaded by then."""

    def __init__(self, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.out_dir = out_dir
        self.calls = 0
        self.existed: list[bool] = []
        self.loaded: list[int] = []
        self._loads = 0
        real = distill_prepare_module.load_kd_rows

        def counting(*args: Any, **kwargs: Any) -> Any:
            self._loads += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(distill_prepare_module, "load_kd_rows", counting)

    def __call__(self) -> None:
        self.calls += 1
        self.existed.append(self.out_dir.exists())
        self.loaded.append(self._loads)


def test_the_hook_runs_once_after_the_inputs_were_read_and_before_anything_is_written(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    hook = Hook(out, monkeypatch)

    prepared = prepare(inputs, out, before_writing=hook)

    assert hook.calls == 1
    assert hook.existed == [False]
    assert hook.loaded == [1]
    assert prepared.written is True
    assert len(files_under(out)) == 4 * 3 * 3 + 3


def test_a_dry_run_never_calls_the_hook(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = Hook(tmp_path / "out", monkeypatch)

    prepared = prepare(inputs, tmp_path / "out", dry_run=True, before_writing=hook)

    assert hook.calls == 0
    assert prepared.written is False
    assert not (tmp_path / "out").exists()


def test_a_refusal_before_the_first_write_never_calls_the_hook(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = Hook(tmp_path / "out", monkeypatch)

    with pytest.raises(ConfigError, match="the manifest vouches for"):
        prepare(
            inputs,
            tmp_path / "out",
            kd_digests={inputs.kd: "a" * 64},
            before_writing=hook,
        )

    assert hook.calls == 0
    assert not (tmp_path / "out").exists()


def test_a_hook_that_raises_stops_the_run_before_it_writes(
    inputs: Inputs, tmp_path: Path
) -> None:
    def refusing() -> None:
        raise OSError("the old manifest cannot be removed")

    with pytest.raises(OSError, match="cannot be removed"):
        prepare(inputs, tmp_path / "out", before_writing=refusing)

    assert not (tmp_path / "out").exists()
