"""Tests for `rinchen spm train` and `rinchen spm check`, the presentation layer
over `rinchen.spm`: what they print, and the exit code each failure maps to.

Models are trained on the tiny synthetic fixture corpus, and every file lands
under pytest's temporary directories only.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import sentencepiece
import yaml
from typer.testing import CliRunner

from rinchen.cli import spm as spm_cli
from rinchen.cli.main import app
from rinchen.config.loader import REPO_ROOT
from rinchen.config.spm import load_spm_config
from rinchen.spm.check import SpmCheckReport, check_spm
from rinchen.spm.symbols import user_defined_symbols
from rinchen.spm.train import (
    MODEL_TYPE,
    NORMALIZATION_RULE_NAME,
    SPLIT_DIGITS,
    train_spm,
)

runner = CliRunner()

TINY_CONFIG_PATH = REPO_ROOT / "configs" / "spm" / "tiny_1k.yaml"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "spm" / "tiny_corpus"
FIXTURE_LANGUAGES = ["eng_Latn", "khk_Cyrl", "mn_Mong", "rus_Cyrl"]

# Same size as the production tiny model, which the fixture corpus supports even
# without the 256 byte pieces.
VARIANT_VOCAB_SIZE = 1000

COVERAGE_LINE = re.compile(r"^(?P<lang>\w+)\t\d\.\d{4}$")
NON_ATOMIC_PREFIX = "non-atomic symbol: "

ArgsBuilder = Callable[[Path, Path], list[str]]


# Far more pieces than the tiny fixture corpus can supply, so the trainer rejects it.
UNREACHABLE_VOCAB_SIZE = 50_000


def _write_tiny_config(
    path: Path,
    *,
    output_prefix: Path,
    corpus_dir: Path = FIXTURE_DIR,
    **overrides: object,
) -> Path:
    """Write the tiny config with its two location fields redirected and any
    further `overrides` applied."""
    fields = yaml.safe_load(TINY_CONFIG_PATH.read_text(encoding="utf-8"))
    fields["corpus_dir"] = str(corpus_dir)
    fields["output_prefix"] = str(output_prefix)
    fields.update(overrides)
    path.write_text(yaml.safe_dump(fields), encoding="utf-8")
    return path


def _train_variant(prefix: Path, *, byte_fallback: bool, symbols: list[str]) -> Path:
    """Train a model the way `train_spm` does, except for the one setting a
    test needs to change."""
    fixture_text = prefix.parent / "fixture_text.txt"
    fixture_text.write_text(
        "".join(
            (FIXTURE_DIR / f"{lang}.txt").read_text(encoding="utf-8")
            for lang in FIXTURE_LANGUAGES
        ),
        encoding="utf-8",
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(fixture_text),
        model_prefix=str(prefix),
        vocab_size=VARIANT_VOCAB_SIZE,
        character_coverage=0.9999,
        model_type=MODEL_TYPE,
        byte_fallback=byte_fallback,
        split_digits=SPLIT_DIGITS,
        normalization_rule_name=NORMALIZATION_RULE_NAME,
        user_defined_symbols=symbols,
        shuffle_input_sentence=True,
        minloglevel=2,
    )
    return Path(f"{prefix}.model")


def _check_args(model: Path, corpus_dir: Path) -> list[str]:
    return ["spm", "check", "--model", str(model), "--corpus-dir", str(corpus_dir)]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = load_spm_config(TINY_CONFIG_PATH).model_copy(
        update={"output_prefix": tmp_path_factory.mktemp("cli_spm") / "tiny"}
    )
    return train_spm(cfg)


def test_train_then_check_round_trip(tmp_path: Path) -> None:
    config = _write_tiny_config(
        tmp_path / "tiny.yaml", output_prefix=tmp_path / "spm" / "tiny"
    )
    model = tmp_path / "spm" / "tiny.model"

    trained = runner.invoke(app, ["spm", "train", "--config", str(config)])

    assert trained.exit_code == 0, trained.output
    assert trained.stdout.splitlines() == [f"trained {model}"]
    assert model.is_file()

    checked = runner.invoke(app, _check_args(model, FIXTURE_DIR))

    assert checked.exit_code == 0, checked.output
    lines = checked.stdout.splitlines()
    coverage_lines = lines[: len(FIXTURE_LANGUAGES)]
    assert [m.group("lang") for m in map(COVERAGE_LINE.match, coverage_lines) if m] == (
        FIXTURE_LANGUAGES
    )
    assert lines[len(FIXTURE_LANGUAGES) :] == ["byte_fallback: on", "ok"]


def test_check_exits_three_for_a_model_without_symbols(tmp_path: Path) -> None:
    model = _train_variant(tmp_path / "no_symbols", byte_fallback=True, symbols=[])

    result = runner.invoke(app, _check_args(model, FIXTURE_DIR))

    assert result.exit_code == 3, result.output
    lines = result.stdout.splitlines()
    assert f"{NON_ATOMIC_PREFIX}<2eng_Latn>" in lines
    listed = [line for line in lines if line.startswith(NON_ATOMIC_PREFIX)]
    assert len(listed) == len(user_defined_symbols())
    assert "byte_fallback: on" in lines
    assert lines[-1] == "failed"


def test_check_exits_three_when_byte_fallback_is_off(tmp_path: Path) -> None:
    model = _train_variant(
        tmp_path / "no_bytes", byte_fallback=False, symbols=user_defined_symbols()
    )

    result = runner.invoke(app, _check_args(model, FIXTURE_DIR))

    assert result.exit_code == 3, result.output
    lines = result.stdout.splitlines()
    assert "byte_fallback: off" in lines
    assert not [line for line in lines if line.startswith(NON_ATOMIC_PREFIX)]
    assert lines[-1] == "failed"


def test_check_forwards_sample_lines(
    tiny_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []

    def recording_check(
        model_path: Path, corpus_files: Mapping[str, Path], *, sample_lines: int
    ) -> SpmCheckReport:
        seen.append(sample_lines)
        return check_spm(model_path, corpus_files, sample_lines=sample_lines)

    monkeypatch.setattr(spm_cli, "check_spm", recording_check)

    result = runner.invoke(
        app, [*_check_args(tiny_model, FIXTURE_DIR), "--sample-lines", "7"]
    )

    assert result.exit_code == 0, result.output
    assert seen == [7]


def test_check_rejects_a_non_positive_sample_count(tiny_model: Path) -> None:
    result = runner.invoke(
        app, [*_check_args(tiny_model, FIXTURE_DIR), "--sample-lines", "0"]
    )

    assert result.exit_code == 2


def _missing_config(workdir: Path, model: Path) -> list[str]:
    return ["spm", "train", "--config", str(workdir / "absent.yaml")]


def _corrupt_corpus_file(workdir: Path, model: Path) -> list[str]:
    corpus_dir = workdir / "corrupt_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "khk_Cyrl.txt").write_bytes(b"\xff\xfe\x00")
    return _check_args(model, corpus_dir)


def _garbage_model_file(workdir: Path, model: Path) -> list[str]:
    garbage = workdir / "garbage.model"
    garbage.write_bytes(b"this is not a sentencepiece model")
    return _check_args(garbage, FIXTURE_DIR)


def _unknown_corpus_stem(workdir: Path, model: Path) -> list[str]:
    corpus_dir = workdir / "unknown_stem_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "foo.txt").write_text("text\n", encoding="utf-8")
    return _check_args(model, corpus_dir)


def _unwritable_output_prefix(workdir: Path, model: Path) -> list[str]:
    blocker = workdir / "blocker"
    blocker.write_text("a regular file, not a directory\n", encoding="utf-8")
    config = _write_tiny_config(
        workdir / "blocked.yaml", output_prefix=blocker / "tiny"
    )
    return ["spm", "train", "--config", str(config)]


def _unreachable_vocab_size(workdir: Path, model: Path) -> list[str]:
    config = _write_tiny_config(
        workdir / "oversized.yaml",
        output_prefix=workdir / "oversized" / "tiny",
        vocab_size=UNREACHABLE_VOCAB_SIZE,
    )
    return ["spm", "train", "--config", str(config)]


@pytest.mark.parametrize(
    ("build_args", "exit_code", "message_start"),
    [
        pytest.param(_missing_config, 2, "error: ", id="missing-config"),
        pytest.param(_corrupt_corpus_file, 3, "error: ", id="corrupt-corpus-file"),
        pytest.param(_garbage_model_file, 3, "error: ", id="garbage-model-file"),
        pytest.param(_unknown_corpus_stem, 2, "error: ", id="unknown-corpus-stem"),
        pytest.param(
            _unwritable_output_prefix,
            1,
            "error: cannot write SentencePiece outputs for ",
            id="unwritable-output-prefix",
        ),
        pytest.param(
            _unreachable_vocab_size,
            4,
            "error: sentencepiece training failed for ",
            id="trainer-rejects-vocab-size",
        ),
    ],
)
def test_domain_errors_exit_with_their_code_and_a_message(
    tmp_path: Path,
    tiny_model: Path,
    build_args: ArgsBuilder,
    exit_code: int,
    message_start: str,
) -> None:
    result = runner.invoke(app, build_args(tmp_path, tiny_model))

    assert result.exit_code == exit_code, result.output
    assert result.stderr.startswith(message_start)
    assert result.stdout == ""


@pytest.mark.parametrize("command", ["train", "check"])
def test_command_help_exits_zero(command: str) -> None:
    result = runner.invoke(app, ["spm", command, "--help"])

    assert result.exit_code == 0
    assert "--help" in result.stdout
