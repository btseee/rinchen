"""`rinchen convert-script`: the five commands, the manifests they write and the
exit code of each failure.

The heavy library calls (`train_char_model`, `check_char_model`, `run_training`,
`export_ct2` and `load_converter`) are module-level names of the CLI; tests replace
them with recorders, so nothing is trained, exported or loaded. Everything lands
under pytest's temporary directories. Mongolian and invisible characters are written
as `chr()` calls; the Cyrillic words are written out.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shlex
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

import pytest
import typer.core
import typer.main
import yaml
from click.testing import Result
from typer.testing import CliRunner

from rinchen.cli import convert_script
from rinchen.cli.main import app
from rinchen.config.loader import REPO_ROOT
from rinchen.errors import DataError, RinchenError, UpstreamError
from rinchen.manifest import store
from rinchen.manifest.model import Manifest
from rinchen.script_conv.alphabet import NNBSP
from rinchen.script_conv.config import (
    render_converter_config,
    validate_converter_config,
)
from rinchen.script_conv.data import build_corpus, read_pairs_tsv, rule_pairs
from rinchen.script_conv.rules import LEXICON, transliterate_word
from rinchen.script_conv.runtime import Conversion, ScriptConverter
from tests.support.script_reference import FIXTURE_DIR
from tests.support.stub_runtime import StubProcessor, StubTranslator

runner = CliRunner()

A = chr(0x1820)
NA = chr(0x1828)
BA = chr(0x182A)
ZWJ = chr(0x200D)
MONGOL = "".join(
    chr(code_point) for code_point in (0x182E, 0x1823, 0x1829, 0x182D, 0x1823, 0x182F)
)
SUFFIX_UN = chr(0x1824) + chr(0x1828)

TINY_WORDS = FIXTURE_DIR / "tiny_words.tsv"
HEADER = "cyrillic\tscript\n"
DATA_STAGE = "script_conv_data"
TRAIN_STAGE = "script_conv_train"
EXPORT_STAGE = "script_conv_export"
CORPUS_NAMES = [
    "cyrl2mong.train.src",
    "cyrl2mong.train.tgt",
    "mong2cyrl.train.src",
    "mong2cyrl.train.tgt",
    "cyrl2mong.valid.src",
    "cyrl2mong.valid.tgt",
    "mong2cyrl.valid.src",
    "mong2cyrl.valid.tgt",
]
CHECKPOINT_FILE = "model.00.safetensors"
CHAR_MODEL_FILE = "converter_char.model"
TRAIN_TOOLS = ["eole", "torch", "sentencepiece", "ctranslate2"]
EXPORT_TOOLS = ["ctranslate2", "eole"]
EOLE = str(Path(sys.executable).with_name("eole"))
COMMANDS = {
    "build-data": [
        "--pairs",
        "--words",
        "--out-dir",
        "--valid-fraction",
        "--seed",
        "--allow-fallback",
    ],
    "train": [
        "--profile",
        "--data-dir",
        "--run-dir",
        "--spm-config",
        "--threads",
        "--dry-run",
        "--confirm-full",
    ],
    "export": ["--run-dir", "--config", "--out-dir", "--timeout"],
    "apply": ["--model-dir", "--target"],
    "eval": ["--model-dir", "--pairs", "--target"],
}


# ---- helpers ------------------------------------------------------------------------


def _run(*args: str, input: bytes | None = None) -> Result:
    return runner.invoke(app, ["convert-script", *args], input=input)


def _run_ok(*args: str, input: bytes | None = None) -> Result:
    result = _run(*args, input=input)
    assert result.exit_code == 0, result.output
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(path: Path) -> list[str]:
    """The lines of a corpus file, split at LF only (NNBSP and U+2028 belong to a
    word)."""
    text = path.read_text(encoding="utf-8")
    return text.removesuffix("\n").split("\n") if text else []


def _manifest(directory: Path) -> Manifest:
    manifest = store.read(directory)
    assert manifest is not None
    return manifest


def _no_part_files(root: Path) -> None:
    assert list(root.rglob("*.part")) == []


def _tree(root: Path) -> list[tuple[str, bytes | None]]:
    """Every path under `root`, with the bytes of each file, to see that nothing
    changed."""
    return [
        (str(path), path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    ]


def _fail_if_called(name: str) -> Callable[..., NoReturn]:
    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError(f"{name} must not be called")

    return fail


def _foreign_manifest() -> Manifest:
    return Manifest(
        stage="mine_align",
        created="2026-11-07T00:00:00Z",
        code_version="deadbeef",
        config_sha256="cfg-abc",
        seed=0,
    )


def _same_stage_manifest(stage: str) -> Manifest:
    return Manifest(
        stage=stage,
        created="2026-11-07T00:00:00Z",
        code_version="deadbeef",
        config_sha256="old",
        seed=0,
    )


# ---- the surface ---------------------------------------------------------------------


def test_the_group_lists_the_five_commands() -> None:
    result = _run("--help")

    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_help_of_each_command_exits_zero_and_lists_its_options(command: str) -> None:
    result = _run(command, "--help")

    assert result.exit_code == 0
    for option in COMMANDS[command]:
        assert option in result.stdout


def test_every_option_and_argument_has_help_text() -> None:
    group = typer.main.get_command(convert_script.app)
    assert isinstance(group, typer.core.TyperGroup)

    assert sorted(group.commands) == sorted(COMMANDS)
    for name, command in group.commands.items():
        assert command.help, f"{name} has no description"
        for param in command.params:
            if param.name != "help":
                assert getattr(param, "help", None), f"{name}: {param.name}"


def test_the_default_locations_are_under_the_repository() -> None:
    assert convert_script.DEFAULT_CORPUS_DIR == (
        REPO_ROOT / "data" / "script_conv" / "corpus"
    )
    assert convert_script.DEFAULT_CHAR_MODEL_CONFIG == (
        REPO_ROOT / "configs" / "spm" / "converter_char.yaml"
    )


# ---- build-data ----------------------------------------------------------------------


@dataclass(frozen=True)
class DataWorkspace:
    pairs: Path
    out_dir: Path
    kept: int  # the pairs that survive screening
    dropped: dict[str, int]  # what the planted bad rows must be counted as

    def args(self, *extra: str) -> list[str]:
        return [
            "build-data",
            "--pairs",
            str(self.pairs),
            "--out-dir",
            str(self.out_dir),
            *extra,
        ]


@pytest.fixture
def data_workspace(tmp_path: Path) -> DataWorkspace:
    """80 reference pairs and five planted rows the screening must drop: two
    exact duplicates, a ZWJ in the script, a Cyrillic letter in the script and a
    space inside the Cyrillic side."""
    rows = TINY_WORDS.read_text(encoding="utf-8").split("\n")[1:81]
    planted = [
        rows[0],
        rows[1],
        f"бүх\t{A}{ZWJ}{NA}",
        f"улс\t{A}а{NA}",
        f"а б\t{A}{NA}",
    ]
    pairs = tmp_path / "input" / "pairs.tsv"
    pairs.parent.mkdir()
    pairs.write_text(HEADER + "\n".join([*rows, *planted]) + "\n", encoding="utf-8")
    return DataWorkspace(
        pairs=pairs,
        out_dir=tmp_path / "corpus",
        kept=80,
        dropped={
            "duplicate": 2,
            "joiner": 1,
            "foreign_character": 1,
            "multiword": 1,
        },
    )


def test_build_data_writes_the_eight_files_and_a_manifest_last(
    data_workspace: DataWorkspace,
) -> None:
    result = _run_ok(*data_workspace.args())

    out_dir = data_workspace.out_dir
    assert {path.name for path in out_dir.iterdir()} == {*CORPUS_NAMES, "manifest.json"}
    assert result.stdout.splitlines()[-1] == f"manifest: {out_dir / 'manifest.json'}"
    manifest = _manifest(out_dir)
    assert manifest.stage == DATA_STAGE
    assert [item.path for item in manifest.outputs] == [
        str(out_dir / name) for name in CORPUS_NAMES
    ]
    for item in manifest.outputs:
        path = Path(item.path)
        assert item.sha256 == _sha256(path)
        assert item.bytes == path.stat().st_size
        assert item.rows == len(_lines(path))
    train_rows = len(_lines(out_dir / "cyrl2mong.train.src"))
    valid_rows = len(_lines(out_dir / "cyrl2mong.valid.src"))
    assert train_rows + valid_rows == data_workspace.kept
    assert valid_rows > 0


def test_build_data_manifest_records_input_counts_and_parameters(
    data_workspace: DataWorkspace,
) -> None:
    _run_ok(*data_workspace.args())

    manifest = _manifest(data_workspace.out_dir)
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (str(data_workspace.pairs), _sha256(data_workspace.pairs))
    ]
    assert manifest.metrics == {
        "valid_fraction": 0.1,
        "pairs_in": 85.0,
        **{f"dropped.{code}": float(n) for code, n in data_workspace.dropped.items()},
    }
    assert manifest.seed == 1234
    canonical = json.dumps(
        {
            "allow_fallback": False,
            "seed": 1234,
            "source": "pairs",
            "valid_fraction": 0.1,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert manifest.config_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest.code_version
    assert datetime.fromisoformat(manifest.created).utcoffset() == timedelta(0)


def test_build_data_writes_what_the_library_builds_from_the_same_pairs(
    data_workspace: DataWorkspace, tmp_path: Path
) -> None:
    _run_ok(*data_workspace.args())

    expected = build_corpus(
        read_pairs_tsv(data_workspace.pairs, origin="parallel"),
        tmp_path / "direct",
        valid_fraction=0.1,
        seed=1234,
    )
    for name in CORPUS_NAMES:
        assert (data_workspace.out_dir / name).read_bytes() == (
            expected.files[name].read_bytes()
        )


def test_build_data_passes_the_seed_and_the_valid_fraction_on(
    data_workspace: DataWorkspace, tmp_path: Path
) -> None:
    _run_ok(*data_workspace.args("--seed", "7", "--valid-fraction", "0.25"))

    manifest = _manifest(data_workspace.out_dir)
    assert manifest.seed == 7
    assert manifest.metrics["valid_fraction"] == 0.25
    expected = build_corpus(
        read_pairs_tsv(data_workspace.pairs, origin="parallel"),
        tmp_path / "direct",
        valid_fraction=0.25,
        seed=7,
    )
    for name in CORPUS_NAMES:
        assert (data_workspace.out_dir / name).read_bytes() == (
            expected.files[name].read_bytes()
        )


def test_build_data_reports_the_dropped_pairs(data_workspace: DataWorkspace) -> None:
    result = _run_ok(*data_workspace.args())

    assert "dropped 5" in result.stdout
    for code in data_workspace.dropped:
        assert code in result.stdout


def test_build_data_default_out_dir_is_the_corpus_dir_of_the_repository(
    data_workspace: DataWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "default_root"
    monkeypatch.setattr(convert_script, "DEFAULT_CORPUS_DIR", default)

    _run_ok("build-data", "--pairs", str(data_workspace.pairs))

    assert (default / "manifest.json").is_file()
    assert not data_workspace.out_dir.exists()


@pytest.fixture
def words_file(tmp_path: Path) -> Path:
    """Twelve lexicon words, a blank line and a word only the fallback can spell."""
    words = sorted(LEXICON)[:12]
    path = tmp_path / "words.txt"
    path.write_text("\n".join([*words, "", "абвгд"]) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("extra", "pairs_in"), [([], 12), (["--allow-fallback"], 13)], ids=["rules", "fb"]
)
def test_build_data_turns_words_into_pairs_through_the_rules(
    words_file: Path,
    tmp_path: Path,
    extra: list[str],
    pairs_in: int,
) -> None:
    out_dir = tmp_path / "corpus"

    _run_ok("build-data", "--words", str(words_file), "--out-dir", str(out_dir), *extra)

    manifest = _manifest(out_dir)
    assert [item.path for item in manifest.inputs] == [str(words_file)]
    assert manifest.metrics["pairs_in"] == float(pairs_in)
    sources = _lines(out_dir / "cyrl2mong.train.src") + _lines(
        out_dir / "cyrl2mong.valid.src"
    )
    assert len(sources) == pairs_in
    assert ("абвгд" in sources) == bool(extra)
    expected_pairs = rule_pairs(
        [*sorted(LEXICON)[:12], "абвгд"], allow_fallback=bool(extra)
    )
    expected = build_corpus(
        expected_pairs, tmp_path / "direct", valid_fraction=0.1, seed=1234
    )
    for name in CORPUS_NAMES:
        assert (out_dir / name).read_bytes() == expected.files[name].read_bytes()
    canonical = json.dumps(
        {
            "allow_fallback": bool(extra),
            "seed": 1234,
            "source": "words",
            "valid_fraction": 0.1,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert manifest.config_sha256 == hashlib.sha256(canonical.encode()).hexdigest()


def test_build_data_says_how_many_words_have_a_rule_spelling(
    words_file: Path, tmp_path: Path
) -> None:
    result = _run_ok(
        "build-data", "--words", str(words_file), "--out-dir", str(tmp_path / "c")
    )

    assert "13 words read, 12 paired" in result.stdout


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--pairs", "PAIRS", "--words", "WORDS"],
        ["--allow-fallback", "--pairs", "P"],
    ],
    ids=["neither", "both", "fallback_with_pairs"],
)
def test_build_data_needs_exactly_one_input_kind(
    data_workspace: DataWorkspace, words_file: Path, options: list[str]
) -> None:
    substitutions = {
        "PAIRS": str(data_workspace.pairs),
        "P": str(data_workspace.pairs),
        "WORDS": str(words_file),
    }
    args = [substitutions.get(option, option) for option in options]

    result = _run("build-data", "--out-dir", str(data_workspace.out_dir), *args)

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("error: ")
    assert "--pairs" in result.stderr or "--allow-fallback" in result.stderr
    assert not data_workspace.out_dir.exists()


def test_build_data_rejects_a_pairs_file_with_another_header(
    data_workspace: DataWorkspace,
) -> None:
    data_workspace.pairs.write_text("script\tcyrillic\nx\ty\n", encoding="utf-8")

    result = _run(*data_workspace.args())

    assert result.exit_code == 3, result.output
    assert "expected header" in result.stderr
    assert str(data_workspace.pairs) in result.stderr
    assert not data_workspace.out_dir.exists()


def test_build_data_rejects_a_missing_input_file(
    data_workspace: DataWorkspace,
) -> None:
    data_workspace.pairs.unlink()

    result = _run(*data_workspace.args())

    assert result.exit_code == 3, result.output
    assert str(data_workspace.pairs) in result.stderr
    assert not data_workspace.out_dir.exists()


def test_build_data_rejects_a_words_file_that_is_not_utf8(
    tmp_path: Path,
) -> None:
    words = tmp_path / "words.txt"
    words.write_bytes(b"\xff\xfe\n")

    result = _run("build-data", "--words", str(words), "--out-dir", str(tmp_path / "c"))

    assert result.exit_code == 3, result.output
    assert "not valid UTF-8" in result.stderr


@pytest.mark.parametrize("spelling", ["same", "dotdot", "symlink"])
def test_build_data_refuses_an_out_dir_that_is_the_directory_of_the_input(
    data_workspace: DataWorkspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
) -> None:
    input_dir = data_workspace.pairs.parent
    out_dir = {
        "same": input_dir,
        "dotdot": input_dir / ".." / input_dir.name,
        "symlink": tmp_path / "alias",
    }[spelling]
    if spelling == "symlink":
        out_dir.symlink_to(input_dir)
    before = _tree(input_dir)
    monkeypatch.setattr(convert_script, "build_corpus", _fail_if_called("build"))
    monkeypatch.setattr(convert_script, "read_pairs_tsv", _fail_if_called("read"))

    result = _run(
        "build-data", "--pairs", str(data_workspace.pairs), "--out-dir", str(out_dir)
    )

    assert result.exit_code == 2, result.output
    assert result.stderr.startswith("error: --out-dir ")
    assert "directory of the input" in result.stderr
    assert _tree(input_dir) == before


def test_build_data_accepts_an_out_dir_beside_the_input_directory(
    data_workspace: DataWorkspace,
) -> None:
    sibling = data_workspace.pairs.parent.with_name("input_corpus")

    _run_ok(
        "build-data", "--pairs", str(data_workspace.pairs), "--out-dir", str(sibling)
    )

    assert (sibling / "manifest.json").is_file()


def test_build_data_refuses_to_replace_the_manifest_of_another_stage_before_reading(
    data_workspace: DataWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.write(data_workspace.out_dir, _foreign_manifest())
    before = _tree(data_workspace.out_dir)
    monkeypatch.setattr(convert_script, "read_pairs_tsv", _fail_if_called("read"))
    monkeypatch.setattr(convert_script, "build_corpus", _fail_if_called("build"))

    result = _run(*data_workspace.args())

    assert result.exit_code == 3, result.output
    assert "mine_align" in result.stderr
    assert f"'{DATA_STAGE}'" in result.stderr
    assert str(data_workspace.out_dir) in result.stderr
    assert _tree(data_workspace.out_dir) == before


def test_build_data_removes_the_old_manifest_before_writing_any_file(
    data_workspace: DataWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ok(*data_workspace.args())
    manifest_present: list[bool] = []
    real_build_corpus = convert_script.build_corpus

    def spy(*args: Any, **kwargs: Any) -> Any:
        manifest_present.append(store.manifest_path(data_workspace.out_dir).exists())
        return real_build_corpus(*args, **kwargs)

    monkeypatch.setattr(convert_script, "build_corpus", spy)

    _run_ok(*data_workspace.args())

    assert manifest_present == [False]
    assert _manifest(data_workspace.out_dir).stage == DATA_STAGE


def test_a_failed_write_leaves_no_manifest_for_a_later_run_to_trust(
    data_workspace: DataWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ok(*data_workspace.args())

    def disk_full(*args: Any, **kwargs: Any) -> NoReturn:
        raise OSError("disk full")

    monkeypatch.setattr(convert_script, "build_corpus", disk_full)

    result = _run(*data_workspace.args())

    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("error: cannot write the corpus files under ")
    assert "disk full" in result.stderr
    assert not store.manifest_path(data_workspace.out_dir).exists()
    _no_part_files(data_workspace.out_dir)


def test_a_run_that_keeps_no_pair_is_a_data_error_that_leaves_no_manifest(
    data_workspace: DataWorkspace,
) -> None:
    _run_ok(*data_workspace.args())
    data_workspace.pairs.write_text(HEADER + f"бүх\t{A}{ZWJ}{NA}\n", encoding="utf-8")

    result = _run(*data_workspace.args())

    assert result.exit_code == 3, result.output
    assert "no pair is left" in result.stderr
    assert not store.manifest_path(data_workspace.out_dir).exists()


def test_a_valid_fraction_the_library_rejects_is_a_config_error(
    data_workspace: DataWorkspace,
) -> None:
    result = _run(*data_workspace.args("--valid-fraction", "0.9"))

    assert result.exit_code == 2, result.output
    assert "valid_fraction" in result.stderr
    assert not store.manifest_path(data_workspace.out_dir).exists()


# ---- train ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainWorkspace:
    data_dir: Path
    run_dir: Path
    spm_config: Path
    char_prefix: Path

    @property
    def char_model(self) -> Path:
        return Path(f"{self.char_prefix}.model")

    @property
    def char_vocab(self) -> Path:
        return Path(f"{self.char_prefix}.vocab")

    @property
    def config(self) -> Path:
        return self.run_dir / "converter.yaml"

    def args(self, *extra: str, profile: str = "tiny") -> list[str]:
        return [
            "train",
            "--profile",
            profile,
            "--data-dir",
            str(self.data_dir),
            "--run-dir",
            str(self.run_dir),
            "--spm-config",
            str(self.spm_config),
            *extra,
        ]

    def corpus_files(self) -> list[Path]:
        return [self.data_dir / name for name in CORPUS_NAMES]

    def expected_texts(self) -> list[str]:
        return [line for path in self.corpus_files() for line in _lines(path)]


@pytest.fixture
def train_workspace(tmp_path: Path) -> TrainWorkspace:
    pairs = read_pairs_tsv(TINY_WORDS, origin="reference")[:60]
    data_dir = tmp_path / "corpus"
    build_corpus(pairs, data_dir, valid_fraction=0.1, seed=1234)
    char_prefix = tmp_path / "char" / "converter_char"
    spm_config = tmp_path / "converter_char.yaml"
    spm_config.write_text(
        yaml.safe_dump({"vocab_size": 200, "output_prefix": str(char_prefix)}),
        encoding="utf-8",
    )
    return TrainWorkspace(
        data_dir=data_dir,
        run_dir=tmp_path / "run",
        spm_config=spm_config,
        char_prefix=char_prefix,
    )


@dataclass
class TrainSeams:
    """Stand-ins for the three heavy calls of `train`. They record the order of the
    calls, whether the manifest existed when each one ran, and the texts they got;
    `fail_at` names the call that raises `error`."""

    workspace: TrainWorkspace
    events: list[str] = field(default_factory=list)
    manifest_present: list[bool] = field(default_factory=list)
    text_types: dict[str, type] = field(default_factory=dict)
    trained_texts: list[str] = field(default_factory=list)
    checked_texts: list[str] = field(default_factory=list)
    configs_at_training: list[dict[str, Any]] = field(default_factory=list)
    vocab_sizes: list[int] = field(default_factory=list)
    threads: list[int] = field(default_factory=list)
    fail_at: str | None = None
    error: RinchenError = field(default_factory=lambda: UpstreamError("boom"))
    writes_checkpoint: bool = True

    def _enter(self, name: str) -> None:
        self.events.append(name)
        self.manifest_present.append(
            store.manifest_path(self.workspace.run_dir).exists()
        )
        if self.fail_at == name:
            raise self.error

    def train_char_model(self, texts: Any, prefix: Path, *, vocab_size: int) -> Path:
        self._enter("train_char_model")
        self.text_types["train"] = type(texts)
        self.trained_texts = list(texts)
        self.vocab_sizes.append(vocab_size)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{prefix}.model").write_bytes(b"char model bytes")
        Path(f"{prefix}.vocab").write_text("<unk>\t0\n", encoding="utf-8")
        return Path(f"{prefix}.model")

    def check_char_model(self, model: Path, texts: Any) -> None:
        self._enter("check_char_model")
        self.text_types["check"] = type(texts)
        self.checked_texts = list(texts)

    def run_training(
        self,
        cfg_path: Path,
        run_dir: Path,
        *,
        threads: int = 2,
        timeout: float | None = None,
    ) -> Path:
        self._enter("run_training")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.configs_at_training.append(cfg)
        self.threads.append(threads)
        checkpoint_dir = Path(cfg["training"]["model_path"])
        if self.writes_checkpoint:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / CHECKPOINT_FILE).write_bytes(b"weights")
        return checkpoint_dir


@pytest.fixture
def seams(
    train_workspace: TrainWorkspace, monkeypatch: pytest.MonkeyPatch
) -> TrainSeams:
    fake = TrainSeams(train_workspace)
    monkeypatch.setattr(convert_script, "train_char_model", fake.train_char_model)
    monkeypatch.setattr(convert_script, "check_char_model", fake.check_char_model)
    monkeypatch.setattr(convert_script, "run_training", fake.run_training)
    return fake


@pytest.fixture
def no_seam_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("train_char_model", "check_char_model", "run_training"):
        monkeypatch.setattr(convert_script, name, _fail_if_called(name))


def test_train_runs_the_char_model_then_its_check_then_the_config_then_training(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    store.write(train_workspace.run_dir, _same_stage_manifest(TRAIN_STAGE))

    result = _run_ok(*train_workspace.args())

    assert seams.events == ["train_char_model", "check_char_model", "run_training"]
    # The old manifest is gone before the first output is touched.
    assert seams.manifest_present == [False, False, False]
    assert _manifest(train_workspace.run_dir).config_sha256 != "old"
    assert result.stdout.splitlines()[-1] == (
        f"manifest: {train_workspace.run_dir / 'manifest.json'}"
    )


def test_train_trains_and_checks_the_char_model_on_the_same_materialised_texts(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    _run_ok(*train_workspace.args())

    expected = train_workspace.expected_texts()
    assert len(expected) > 200
    assert seams.text_types == {"train": list, "check": list}
    assert seams.trained_texts == expected
    assert seams.checked_texts == expected
    assert seams.vocab_sizes == [200]


def test_train_reads_every_side_of_every_corpus_file_but_no_blank_line(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    for name in ("cyrl2mong.valid.src", "cyrl2mong.valid.tgt"):
        (train_workspace.data_dir / name).write_text("", encoding="utf-8")

    _run_ok(*train_workspace.args())

    assert "" not in seams.trained_texts
    assert seams.trained_texts == train_workspace.expected_texts()


def test_train_writes_the_converter_config_before_it_trains(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    _run_ok(*train_workspace.args())

    expected = render_converter_config(
        "tiny",
        data_dir=train_workspace.data_dir,
        spm_model=train_workspace.char_model,
        run_dir=train_workspace.run_dir,
    )
    validate_converter_config(expected, profile="tiny")
    assert seams.configs_at_training == [expected]
    assert (
        yaml.safe_load(train_workspace.config.read_text(encoding="utf-8")) == expected
    )


def test_train_passes_the_threads_on_and_defaults_to_two(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    _run_ok(*train_workspace.args())
    _run_ok(*train_workspace.args("--threads", "3"))

    assert seams.threads == [2, 3]


def test_train_rejects_fewer_than_one_thread(
    train_workspace: TrainWorkspace, no_seam_runs: None
) -> None:
    result = _run(*train_workspace.args("--threads", "0"))

    assert result.exit_code == 2, result.output


def test_train_manifest_records_inputs_outputs_and_environment(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    _run_ok(*train_workspace.args("--threads", "3"))

    manifest = _manifest(train_workspace.run_dir)
    checkpoint = train_workspace.run_dir / "ckpt" / CHECKPOINT_FILE
    assert manifest.stage == TRAIN_STAGE
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (str(path), _sha256(path))
        for path in (*train_workspace.corpus_files(), train_workspace.spm_config)
    ]
    assert [(item.path, item.sha256, item.bytes) for item in manifest.outputs] == [
        (str(path), _sha256(path), path.stat().st_size)
        for path in (
            train_workspace.char_model,
            train_workspace.char_vocab,
            train_workspace.config,
            checkpoint,
        )
    ]
    assert manifest.config_sha256 == _sha256(train_workspace.config)
    assert manifest.seed == 1234
    assert manifest.metrics == {"threads": 3.0}
    assert manifest.status == {"profile": "tiny"}
    assert list(manifest.tool_versions) == TRAIN_TOOLS
    assert manifest.tool_versions["eole"] == importlib.metadata.version("eole")
    assert manifest.code_version
    assert datetime.fromisoformat(manifest.created).utcoffset() == timedelta(0)


def test_train_manifest_records_both_hashes_of_the_char_model(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    _run_ok(*train_workspace.args())

    recorded = {
        item.path: item.sha256 for item in _manifest(train_workspace.run_dir).outputs
    }
    model_hash = hashlib.sha256(b"char model bytes").hexdigest()
    vocab_hash = hashlib.sha256(b"<unk>\t0\n").hexdigest()
    assert recorded[str(train_workspace.char_model)] == model_hash
    assert recorded[str(train_workspace.char_vocab)] == vocab_hash


def test_train_with_the_full_profile_renders_the_full_config(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    _run_ok(*train_workspace.args("--confirm-full", profile="full"))

    expected = render_converter_config(
        "full",
        data_dir=train_workspace.data_dir,
        spm_model=train_workspace.char_model,
        run_dir=train_workspace.run_dir,
    )
    assert seams.configs_at_training == [expected]
    assert _manifest(train_workspace.run_dir).status == {"profile": "full"}


def test_train_full_profile_is_an_operator_step_without_confirmation(
    train_workspace: TrainWorkspace, no_seam_runs: None
) -> None:
    result = _run(*train_workspace.args(profile="full"))

    assert result.exit_code == 2, result.output
    assert "--confirm-full" in result.stderr
    assert "operator" in result.stderr
    assert not train_workspace.run_dir.exists()
    assert not train_workspace.char_prefix.parent.exists()


def test_train_rejects_an_unknown_profile(
    train_workspace: TrainWorkspace, no_seam_runs: None
) -> None:
    result = _run(*train_workspace.args(profile="huge"))

    assert result.exit_code == 2, result.output
    assert "huge" in result.stderr


def test_train_rejects_a_missing_corpus_file_before_any_work(
    train_workspace: TrainWorkspace, no_seam_runs: None
) -> None:
    (train_workspace.data_dir / "mong2cyrl.valid.tgt").unlink()

    result = _run(*train_workspace.args())

    assert result.exit_code == 3, result.output
    assert "mong2cyrl.valid.tgt" in result.stderr
    assert not train_workspace.run_dir.exists()


def test_train_rejects_a_missing_char_model_config(
    train_workspace: TrainWorkspace, no_seam_runs: None
) -> None:
    train_workspace.spm_config.unlink()

    result = _run(*train_workspace.args())

    assert result.exit_code == 2, result.output
    assert "not found" in result.stderr


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_train_refuses_to_replace_the_manifest_of_another_stage(
    train_workspace: TrainWorkspace, no_seam_runs: None, extra: list[str]
) -> None:
    store.write(train_workspace.run_dir, _foreign_manifest())
    before = _tree(train_workspace.run_dir)

    result = _run(*train_workspace.args(*extra))

    assert result.exit_code == 3, result.output
    assert "mine_align" in result.stderr
    assert f"'{TRAIN_STAGE}'" in result.stderr
    assert _tree(train_workspace.run_dir) == before


def test_train_dry_run_prints_the_model_the_config_and_the_two_commands(
    train_workspace: TrainWorkspace, no_seam_runs: None
) -> None:
    config = train_workspace.config

    result = _run_ok(*train_workspace.args("--dry-run"))

    assert result.stdout.splitlines() == [
        str(train_workspace.char_model),
        str(config),
        f"{EOLE} build_vocab -config {config} -n_sample -1",
        f"{EOLE} train -config {config}",
    ]
    assert result.stderr == ""


def test_train_dry_run_writes_and_runs_nothing(
    train_workspace: TrainWorkspace, tmp_path: Path, no_seam_runs: None
) -> None:
    before = _tree(tmp_path)

    _run_ok(*train_workspace.args("--dry-run"))

    assert _tree(tmp_path) == before
    assert not train_workspace.run_dir.exists()
    assert not train_workspace.char_prefix.parent.exists()


def test_train_dry_run_defaults_to_the_repository_char_model_config(
    train_workspace: TrainWorkspace, no_seam_runs: None
) -> None:
    args = [
        "train",
        "--profile",
        "tiny",
        "--data-dir",
        str(train_workspace.data_dir),
        "--run-dir",
        str(train_workspace.run_dir),
        "--dry-run",
    ]

    result = _run_ok(*args)

    assert result.stdout.splitlines()[0] == str(
        REPO_ROOT / "data" / "script_conv" / "converter_char.model"
    )


def test_train_dry_run_of_the_full_profile_needs_no_confirmation(
    train_workspace: TrainWorkspace, no_seam_runs: None
) -> None:
    result = _run_ok(*train_workspace.args("--dry-run", profile="full"))

    assert len(result.stdout.splitlines()) == 4


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_train_dry_run_checks_the_corpus_files_like_a_real_run(
    train_workspace: TrainWorkspace, no_seam_runs: None, extra: list[str]
) -> None:
    (train_workspace.data_dir / "cyrl2mong.train.src").unlink()

    result = _run(*train_workspace.args(*extra))

    assert result.exit_code == 3, result.output
    assert "cyrl2mong.train.src" in result.stderr


def test_train_dry_run_uses_the_command_builder_of_the_library(
    train_workspace: TrainWorkspace,
    no_seam_runs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = (("one", "a b"), ("two",))
    monkeypatch.setattr(convert_script, "eole_commands", lambda cfg_path: commands)

    result = _run_ok(*train_workspace.args("--dry-run"))

    assert result.stdout.splitlines()[2:] == [shlex.join(c) for c in commands]


@pytest.mark.parametrize(
    ("fail_at", "error", "exit_code", "events"),
    [
        ("train_char_model", UpstreamError("sentencepiece failed"), 4, 1),
        ("check_char_model", DataError("character U+0041 is unknown"), 3, 2),
        ("run_training", UpstreamError("eole train failed"), 4, 3),
    ],
)
def test_a_failing_step_of_train_leaves_no_manifest_and_stops_there(
    train_workspace: TrainWorkspace,
    seams: TrainSeams,
    fail_at: str,
    error: RinchenError,
    exit_code: int,
    events: int,
) -> None:
    seams.fail_at = fail_at
    seams.error = error

    result = _run(*train_workspace.args())

    assert result.exit_code == exit_code, result.output
    assert result.stderr == f"error: {error}\n"
    assert len(seams.events) == events
    assert not store.manifest_path(train_workspace.run_dir).exists()
    _no_part_files(train_workspace.run_dir.parent)
    # The config is written once the char model has passed its check.
    assert train_workspace.config.exists() == (fail_at == "run_training")


def test_a_failed_rerun_removes_the_manifest_of_the_earlier_run(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    _run_ok(*train_workspace.args())
    seams.fail_at = "run_training"

    result = _run(*train_workspace.args())

    assert result.exit_code == 4, result.output
    assert not store.manifest_path(train_workspace.run_dir).exists()


def test_a_training_that_leaves_no_checkpoint_is_a_failure_without_a_manifest(
    train_workspace: TrainWorkspace, seams: TrainSeams
) -> None:
    seams.writes_checkpoint = False

    result = _run(*train_workspace.args())

    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("error: cannot write the training outputs under ")
    assert not store.manifest_path(train_workspace.run_dir).exists()
    _no_part_files(train_workspace.run_dir.parent)


# ---- export --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportWorkspace:
    run_dir: Path
    config: Path
    checkpoint_dir: Path
    char_model: Path

    @property
    def checkpoint(self) -> Path:
        return self.checkpoint_dir / CHECKPOINT_FILE

    @property
    def default_out_dir(self) -> Path:
        return self.run_dir / "ct2"

    def args(self, *extra: str) -> list[str]:
        return ["export", "--run-dir", str(self.run_dir), *extra]


@pytest.fixture
def export_workspace(tmp_path: Path) -> ExportWorkspace:
    run_dir = tmp_path / "run"
    char_model = tmp_path / "char" / "converter_char.model"
    char_model.parent.mkdir()
    char_model.write_bytes(b"char model bytes")
    cfg = render_converter_config(
        "tiny", data_dir=tmp_path / "corpus", spm_model=char_model, run_dir=run_dir
    )
    config = run_dir / "converter.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    checkpoint_dir = run_dir / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / CHECKPOINT_FILE).write_bytes(b"weights")
    return ExportWorkspace(
        run_dir=run_dir,
        config=config,
        checkpoint_dir=checkpoint_dir,
        char_model=char_model,
    )


@dataclass
class ExportSeam:
    """Stands in for `export_ct2`: creates the model directory with three files, as
    the converter does, and records its arguments."""

    calls: list[tuple[Path, Path, float | None]] = field(default_factory=list)
    error: RinchenError | None = None
    blocks_the_char_model: bool = False

    def __call__(
        self, ckpt_dir: Path, out_dir: Path, *, timeout: float | None = None
    ) -> Path:
        self.calls.append((ckpt_dir, out_dir, timeout))
        out_dir.mkdir(parents=True)
        (out_dir / "model.bin").write_bytes(b"ct2 weights")
        if self.error is not None:
            raise self.error
        (out_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (out_dir / "shared_vocabulary.json").write_text("[]\n", encoding="utf-8")
        if self.blocks_the_char_model:
            (out_dir / CHAR_MODEL_FILE).mkdir()
        return out_dir


@pytest.fixture
def export_seam(monkeypatch: pytest.MonkeyPatch) -> ExportSeam:
    seam = ExportSeam()
    monkeypatch.setattr(convert_script, "export_ct2", seam)
    return seam


def test_export_converts_the_checkpoint_the_config_names(
    export_workspace: ExportWorkspace, export_seam: ExportSeam
) -> None:
    result = _run_ok(*export_workspace.args())

    out_dir = export_workspace.default_out_dir
    assert export_seam.calls == [(export_workspace.checkpoint_dir, out_dir, None)]
    assert result.stdout.splitlines()[-1] == f"manifest: {out_dir / 'manifest.json'}"


def test_export_puts_the_char_model_into_the_model_directory(
    export_workspace: ExportWorkspace, export_seam: ExportSeam
) -> None:
    _run_ok(*export_workspace.args())

    copy = export_workspace.default_out_dir / CHAR_MODEL_FILE
    assert copy.read_bytes() == export_workspace.char_model.read_bytes()


def test_export_manifest_lists_the_inputs_and_every_file_of_the_model_directory(
    export_workspace: ExportWorkspace, export_seam: ExportSeam
) -> None:
    _run_ok(*export_workspace.args())

    out_dir = export_workspace.default_out_dir
    manifest = _manifest(out_dir)
    assert manifest.stage == EXPORT_STAGE
    assert [(item.path, item.sha256) for item in manifest.inputs] == [
        (str(path), _sha256(path))
        for path in (export_workspace.checkpoint, export_workspace.char_model)
    ]
    names = ["config.json", CHAR_MODEL_FILE, "model.bin", "shared_vocabulary.json"]
    assert [(item.path, item.sha256, item.bytes) for item in manifest.outputs] == [
        (str(out_dir / name), _sha256(out_dir / name), (out_dir / name).stat().st_size)
        for name in names
    ]
    assert manifest.config_sha256 == _sha256(export_workspace.config)
    assert list(manifest.tool_versions) == EXPORT_TOOLS
    assert manifest.tool_versions["ctranslate2"] == importlib.metadata.version(
        "ctranslate2"
    )
    assert manifest.code_version
    _no_part_files(export_workspace.run_dir)


def test_export_takes_the_config_and_the_out_dir_from_its_options(
    export_workspace: ExportWorkspace, export_seam: ExportSeam, tmp_path: Path
) -> None:
    moved = tmp_path / "elsewhere" / "settings.yaml"
    moved.parent.mkdir()
    export_workspace.config.rename(moved)
    out_dir = tmp_path / "model"

    _run_ok(*export_workspace.args("--config", str(moved), "--out-dir", str(out_dir)))

    assert export_seam.calls == [(export_workspace.checkpoint_dir, out_dir, None)]
    assert (out_dir / "manifest.json").is_file()
    assert not export_workspace.default_out_dir.exists()
    assert _manifest(out_dir).config_sha256 == _sha256(moved)


def test_export_passes_the_timeout_on(
    export_workspace: ExportWorkspace, export_seam: ExportSeam
) -> None:
    _run_ok(*export_workspace.args("--timeout", "12.5"))

    assert export_seam.calls[0][2] == 12.5


def test_export_refuses_an_out_dir_that_exists_and_touches_nothing(
    export_workspace: ExportWorkspace, export_seam: ExportSeam
) -> None:
    out_dir = export_workspace.default_out_dir
    out_dir.mkdir()
    (out_dir / "earlier.bin").write_bytes(b"earlier")
    before = _tree(export_workspace.run_dir)

    result = _run(*export_workspace.args())

    assert result.exit_code == 2, result.output
    assert str(out_dir) in result.stderr
    assert "remove" in result.stderr
    assert export_seam.calls == []
    assert _tree(export_workspace.run_dir) == before


def test_a_failed_export_leaves_no_manifest_and_no_part_file(
    export_workspace: ExportWorkspace, export_seam: ExportSeam
) -> None:
    export_seam.error = UpstreamError("converter exited with code 1")

    result = _run(*export_workspace.args())

    assert result.exit_code == 4, result.output
    assert result.stderr == "error: converter exited with code 1\n"
    assert not store.manifest_path(export_workspace.default_out_dir).exists()
    assert not (export_workspace.default_out_dir / CHAR_MODEL_FILE).exists()
    _no_part_files(export_workspace.run_dir)


def test_a_failed_copy_of_the_char_model_leaves_no_manifest_and_no_part_file(
    export_workspace: ExportWorkspace, export_seam: ExportSeam
) -> None:
    export_seam.blocks_the_char_model = True

    result = _run(*export_workspace.args())

    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("error: cannot write the export outputs under ")
    assert not store.manifest_path(export_workspace.default_out_dir).exists()
    _no_part_files(export_workspace.run_dir)


def test_export_rejects_a_missing_config(
    export_workspace: ExportWorkspace, export_seam: ExportSeam
) -> None:
    export_workspace.config.unlink()

    result = _run(*export_workspace.args())

    assert result.exit_code == 2, result.output
    assert "not found" in result.stderr
    assert export_seam.calls == []


@pytest.mark.parametrize("missing", ["checkpoint", "char_model"])
def test_export_rejects_a_missing_checkpoint_or_char_model_before_it_converts(
    export_workspace: ExportWorkspace, export_seam: ExportSeam, missing: str
) -> None:
    gone = {
        "checkpoint": export_workspace.checkpoint,
        "char_model": export_workspace.char_model,
    }[missing]
    gone.unlink()

    result = _run(*export_workspace.args())

    assert result.exit_code == 3, result.output
    assert str(gone) in result.stderr
    assert export_seam.calls == []
    assert not export_workspace.default_out_dir.exists()


# ---- apply ---------------------------------------------------------------------------


class StubConverter:
    """Answers every text with `answer` and records what it was asked."""

    def __init__(self, answer: Conversion) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def convert(self, text: str, target: str) -> Conversion:
        self.calls.append((text, target))
        return self.answer


@dataclass
class LoadSeam:
    converter: Any
    loaded: list[Path] = field(default_factory=list)

    def __call__(self, model_dir: Path) -> Any:
        self.loaded.append(model_dir)
        return self.converter


def _load_seam(monkeypatch: pytest.MonkeyPatch, converter: Any) -> LoadSeam:
    seam = LoadSeam(converter)
    monkeypatch.setattr(convert_script, "load_converter", seam)
    return seam


def test_apply_converts_stdin_and_writes_the_text_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    converted = f"{MONGOL}{NNBSP}{SUFFIX_UN} {A}{NA}"
    converter = StubConverter(Conversion(converted, 2, 1))
    seam = _load_seam(monkeypatch, converter)
    source = "монгол улс\n".encode()

    result = _run_ok(
        "apply", "--model-dir", str(tmp_path / "m"), "--target", "Mong", input=source
    )

    assert result.stdout_bytes == converted.encode("utf-8")
    assert b"\xe2\x80\xaf" in result.stdout_bytes
    assert result.stderr == "rule_fallbacks=2 invalid_outputs=1\n"
    assert seam.loaded == [tmp_path / "m"]
    # The whole input is one text; its newline goes with it.
    assert converter.calls == [("монгол улс\n", "Mong")]


def test_apply_converts_the_whole_file_as_one_text_and_keeps_line_ends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    converter = StubConverter(Conversion("out", 0, 0))
    _load_seam(monkeypatch, converter)
    source = tmp_path / "in.txt"
    source.write_bytes("а\r\nб\n".encode())

    result = _run_ok(
        "apply", "--model-dir", str(tmp_path), "--target", "Cyrl", str(source)
    )

    assert converter.calls == [("а\r\nб\n", "Cyrl")]
    assert result.stdout_bytes == b"out"


def test_apply_passes_whitespace_only_text_to_the_converter_and_writes_its_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    converter = StubConverter(Conversion("  \n", 0, 0))
    _load_seam(monkeypatch, converter)

    result = _run_ok(
        "apply", "--model-dir", str(tmp_path), "--target", "Mong", input=b"  \n"
    )

    assert converter.calls == [("  \n", "Mong")]
    assert result.stdout_bytes == b"  \n"


@pytest.mark.parametrize("from_file", [False, True], ids=["stdin", "file"])
def test_apply_of_empty_input_writes_nothing_and_loads_no_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, from_file: bool
) -> None:
    monkeypatch.setattr(convert_script, "load_converter", _fail_if_called("load"))
    args = ["apply", "--model-dir", str(tmp_path), "--target", "Mong"]
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")

    result = _run_ok(*args, *([str(source)] if from_file else []), input=b"")

    assert result.stdout_bytes == b""
    assert result.stderr == ""


@pytest.mark.parametrize("target", ["Latn", "mong", ""])
def test_apply_rejects_an_unknown_target_before_it_reads_or_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    monkeypatch.setattr(convert_script, "load_converter", _fail_if_called("load"))

    result = _run(
        "apply", "--model-dir", str(tmp_path), "--target", target, input=b"text"
    )

    assert result.exit_code == 2, result.output
    assert "Mong" in result.stderr
    assert "Cyrl" in result.stderr
    assert result.stdout_bytes == b""


def test_apply_reports_a_missing_input_file_as_a_data_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(convert_script, "load_converter", _fail_if_called("load"))
    absent = tmp_path / "absent.txt"

    result = _run(
        "apply", "--model-dir", str(tmp_path), "--target", "Mong", str(absent)
    )

    assert result.exit_code == 3, result.output
    assert str(absent) in result.stderr


@pytest.mark.parametrize("from_file", [False, True], ids=["stdin", "file"])
def test_apply_reports_input_that_is_not_utf8_as_a_data_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, from_file: bool
) -> None:
    monkeypatch.setattr(convert_script, "load_converter", _fail_if_called("load"))
    source = tmp_path / "bad.txt"
    source.write_bytes(b"\xff\xfe")
    args = ["apply", "--model-dir", str(tmp_path), "--target", "Mong"]

    result = _run(*args, *([str(source)] if from_file else []), input=b"\xff\xfe")

    assert result.exit_code == 3, result.output
    assert "UTF-8" in result.stderr


def test_apply_reports_a_model_that_cannot_be_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def cannot_load(model_dir: Path) -> NoReturn:
        raise DataError(f"cannot load the CTranslate2 model in {model_dir}")

    monkeypatch.setattr(convert_script, "load_converter", cannot_load)

    result = _run(
        "apply", "--model-dir", str(tmp_path), "--target", "Mong", input=b"text"
    )

    assert result.exit_code == 3, result.output
    assert result.stderr.startswith("error: cannot load the CTranslate2 model")


def test_apply_through_the_real_converter_spells_a_fallback_word_and_counts_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answers = {"монгол": MONGOL, "улс": A + ZWJ}
    translator = StubTranslator(answers.__getitem__)
    converter = ScriptConverter.from_parts(translator, StubProcessor())
    _load_seam(monkeypatch, converter)
    rule_word = transliterate_word("улс").text

    result = _run_ok(
        "apply",
        "--model-dir",
        str(tmp_path),
        "--target",
        "Mong",
        input="монгол улс".encode(),
    )

    assert result.stdout_bytes == f"{MONGOL} {rule_word}".encode()
    assert result.stderr == "rule_fallbacks=1 invalid_outputs=0\n"


def test_apply_through_the_real_converter_spells_a_word_too_long_for_the_model_by_rules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    translator = StubTranslator({}.__getitem__)
    converter = ScriptConverter.from_parts(translator, StubProcessor())
    _load_seam(monkeypatch, converter)
    long_word = "а" * 81

    result = _run_ok(
        "apply",
        "--model-dir",
        str(tmp_path),
        "--target",
        "Mong",
        input=long_word.encode(),
    )

    assert result.stdout_bytes == transliterate_word(long_word).text.encode()
    assert result.stderr == "rule_fallbacks=1 invalid_outputs=0\n"
    assert translator.calls == []


@pytest.mark.parametrize("text", [" ", "\n", " \t\r\n "], ids=repr)
def test_apply_through_the_real_converter_returns_whitespace_only_text_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str
) -> None:
    translator = StubTranslator({}.__getitem__)
    converter = ScriptConverter.from_parts(translator, StubProcessor())
    _load_seam(monkeypatch, converter)

    result = _run_ok(
        "apply",
        "--model-dir",
        str(tmp_path),
        "--target",
        "Mong",
        input=text.encode(),
    )

    assert result.stdout_bytes == text.encode()
    assert result.stderr == "rule_fallbacks=0 invalid_outputs=0\n"
    assert translator.calls == []


def test_load_converter_loads_the_char_model_that_sits_in_the_model_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: list[tuple[Path, Path]] = []

    class RecordingConverter:
        def __init__(self, ct2_dir: Path, spm_model: Path) -> None:
            loaded.append((ct2_dir, spm_model))

    monkeypatch.setattr(convert_script, "ScriptConverter", RecordingConverter)

    convert_script.load_converter(tmp_path / "model")

    assert loaded == [(tmp_path / "model", tmp_path / "model" / CHAR_MODEL_FILE)]


# ---- eval ----------------------------------------------------------------------------


class MappingConverter:
    """Converts a word by lookup and records every call."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def convert(self, text: str, target: str) -> Conversion:
        self.calls.append((text, target))
        return Conversion(self.answers[text], 0, 0)


PAIRS = [
    ("ана", A + NA + A),
    ("аба", A + BA + A),
    ("нан", NA + A + NA),
    ("ааа", A + A + A),
]


def _pairs_file(tmp_path: Path, pairs: Sequence[tuple[str, str]] = PAIRS) -> Path:
    path = tmp_path / "eval.tsv"
    path.write_text(HEADER + "".join(f"{c}\t{s}\n" for c, s in pairs), encoding="utf-8")
    return path


def test_eval_scores_the_conversion_of_the_cyrillic_side_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One substituted letter in the second word and one dropped from the last: two
    # edits over 12 reference characters, and two of four words exact.
    answers = {
        "ана": A + NA + A,
        "аба": A + NA + A,
        "нан": NA + A + NA,
        "ааа": A + A,
    }
    converter = MappingConverter(answers)
    seam = _load_seam(monkeypatch, converter)

    result = _run_ok(
        "eval",
        "--model-dir",
        str(tmp_path / "m"),
        "--pairs",
        str(_pairs_file(tmp_path)),
    )

    assert result.stdout.splitlines() == ["cer 0.1667", "word_accuracy 0.5000", "n 4"]
    assert seam.loaded == [tmp_path / "m"]
    assert converter.calls == [(cyrillic, "Mong") for cyrillic, _ in PAIRS]


def test_eval_of_the_cyrillic_direction_converts_the_script_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answers = {script: cyrillic for cyrillic, script in PAIRS}
    answers[PAIRS[0][1]] = "ан"
    converter = MappingConverter(answers)
    _load_seam(monkeypatch, converter)

    result = _run_ok(
        "eval",
        "--model-dir",
        str(tmp_path),
        "--pairs",
        str(_pairs_file(tmp_path)),
        "--target",
        "Cyrl",
    )

    assert result.stdout.splitlines() == ["cer 0.0833", "word_accuracy 0.7500", "n 4"]
    assert converter.calls == [(script, "Cyrl") for _, script in PAIRS]


def test_eval_rejects_an_unknown_target_before_it_loads_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(convert_script, "load_converter", _fail_if_called("load"))

    result = _run(
        "eval",
        "--model-dir",
        str(tmp_path),
        "--pairs",
        str(_pairs_file(tmp_path)),
        "--target",
        "Latn",
    )

    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("content", [HEADER, ""], ids=["header_only", "empty_file"])
def test_eval_of_a_file_without_pairs_is_a_data_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    monkeypatch.setattr(convert_script, "load_converter", _fail_if_called("load"))
    pairs = tmp_path / "eval.tsv"
    pairs.write_text(content, encoding="utf-8")

    result = _run("eval", "--model-dir", str(tmp_path), "--pairs", str(pairs))

    assert result.exit_code == 3, result.output
    assert str(pairs) in result.stderr
    assert result.stdout == ""


def test_eval_rejects_a_missing_pairs_file_and_a_wrong_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(convert_script, "load_converter", _fail_if_called("load"))
    absent = tmp_path / "absent.tsv"
    wrong = tmp_path / "wrong.tsv"
    wrong.write_text("script\tcyrillic\nx\ty\n", encoding="utf-8")

    for path in (absent, wrong):
        result = _run("eval", "--model-dir", str(tmp_path), "--pairs", str(path))

        assert result.exit_code == 3, result.output
        assert str(path) in result.stderr
