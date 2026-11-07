"""Tests for `rinchen.script_conv.train`: the Eole subprocesses of the converter.

No test starts Eole. `subprocess.Popen` is replaced by a recorder that writes what
the real tool would (log lines, the checkpoint files, the prediction file), so each
test pins the commands, the environment and the failure handling. The runner's own
process handling (timeouts, killing a process group) is tested with real children in
`test_script_conv_subprocess.py`; the real Eole run is the integration test in
`tests/integration/test_script_conv_tiny.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import sentencepiece
import yaml

from rinchen.errors import ConfigError, DataError, UpstreamError
from rinchen.script_conv import train as train_module
from rinchen.script_conv.char_model import train_char_model
from rinchen.script_conv.config import dump_config, render_converter_config
from rinchen.script_conv.train import (
    eole_commands,
    predict,
    predict_side_paths,
    run_training,
    training_environment,
)
from tests.support.fake_process import FakeProcess

TO_SCRIPT_TAG = "<2mn_Mong>"
TO_CYRILLIC_TAG = "<2mn_Cyrl>"
DUMMY_PREFIX = chr(0x2581)  # SentencePiece's dummy prefix, a piece of its own
NNBSP = chr(0x202F)
EOLE_NAMES = ("eole", "eole.exe")
CHECKPOINT_FILE = "model.00.safetensors"  # what Eole links to its newest checkpoint


class FakeEole:
    """Stands in for `subprocess.Popen`: records every launch, writes `output` lines
    to the log handle it is given and exits with `exit_codes[subcommand]` (default 0).
    A finished `train` leaves a checkpoint (Eole's layout: a step directory and a
    link to its file), a finished `predict` its raw token output file, unless
    `writes_results` is false. `checkpoint_age` is how many seconds before now the
    written checkpoint's modification time is set.

    Every `predict` launch records the prediction config and the lines of the token
    source file it names in `predict_inputs`. The raw output is `hypotheses` (one
    token line each) or, when that is None, the source line without its tag: a
    model that copies its input."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.waits: list[float | None] = []
        self.exit_codes: dict[str, int] = {}
        self.output: Sequence[str] = ("a log line",)
        self.writes_results = True
        self.checkpoint_age = 0.0
        self.hypotheses: list[str] | None = None
        self.predict_inputs: list[tuple[dict[str, Any], list[str]]] = []

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((list(argv), kwargs))
        subcommand = argv[1]
        if subcommand == "predict":
            self._record_predict_input(Path(argv[3]))
        for line in self.output:
            kwargs["stdout"].write(f"{line}\n".encode())
        code = self.exit_codes.get(subcommand, 0)
        if code == 0 and self.writes_results:
            self._leave_results(subcommand, Path(argv[3]))
        return FakeProcess(code, self.waits)

    def _leave_results(self, subcommand: str, config_path: Path) -> None:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if subcommand == "train":
            self._write_checkpoint(Path(cfg["training"]["model_path"]))
        elif subcommand == "predict":
            source_lines = self.predict_inputs[-1][1]
            hypotheses = self.hypotheses
            if hypotheses is None:
                hypotheses = [" ".join(line.split(" ")[1:]) for line in source_lines]
            Path(cfg["output"]).write_text(
                "".join(f"{hypothesis}\n" for hypothesis in hypotheses),
                encoding="utf-8",
            )

    def _record_predict_input(self, config_path: Path) -> None:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        source = Path(cfg["src"]).read_text(encoding="utf-8")
        self.predict_inputs.append((cfg, source.removesuffix("\n").split("\n")))

    def _write_checkpoint(self, model_path: Path) -> None:
        step_dir = model_path / "step_1"
        step_dir.mkdir(parents=True, exist_ok=True)
        weights = step_dir / CHECKPOINT_FILE
        weights.write_bytes(b"weights")
        written = time.time() - self.checkpoint_age
        os.utime(weights, (written, written))
        link = model_path / CHECKPOINT_FILE
        link.unlink(missing_ok=True)
        link.symlink_to(Path("step_1") / CHECKPOINT_FILE)

    @property
    def subcommands(self) -> list[str]:
        return [argv[1] for argv, _ in self.calls]


@pytest.fixture
def fake_eole(monkeypatch: pytest.MonkeyPatch) -> FakeEole:
    fake = FakeEole()
    monkeypatch.setattr(subprocess, "Popen", fake)
    return fake


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "run"


def _write_config(
    tmp_path: Path, run_dir: Path, *, profile: str = "tiny"
) -> tuple[Path, dict[str, Any]]:
    cfg = render_converter_config(
        profile,  # type: ignore[arg-type]
        data_dir=tmp_path / "corpus",
        spm_model=tmp_path / "converter_char.model",
        run_dir=run_dir,
    )
    cfg_path = tmp_path / "converter.yaml"
    dump_config(cfg, cfg_path)
    return cfg_path, cfg


@pytest.fixture
def tiny_config(tmp_path: Path, run_dir: Path) -> Path:
    return _write_config(tmp_path, run_dir)[0]


def _stale_checkpoint(model_path: Path, *, age_seconds: float = 3600.0) -> Path:
    """A checkpoint file an earlier run left, last written `age_seconds` ago."""
    model_path.mkdir(parents=True)
    stale = model_path / CHECKPOINT_FILE
    stale.write_bytes(b"old weights")
    written = time.time() - age_seconds
    os.utime(stale, (written, written))
    return stale


# ---- training_environment -------------------------------------------------------


def test_training_environment_sets_both_thread_variables_as_strings_and_keeps_path() -> (
    None
):
    env = training_environment(3, hide_gpu=False)

    assert env["OMP_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] == "3"
    assert env["PATH"] == os.environ["PATH"]


def test_training_environment_hides_the_gpu_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert training_environment(2, hide_gpu=True)["CUDA_VISIBLE_DEVICES"] == ""
    assert training_environment(2, hide_gpu=False)["CUDA_VISIBLE_DEVICES"] == "0"


def test_training_environment_leaves_an_unset_gpu_variable_unset_when_not_hiding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert "CUDA_VISIBLE_DEVICES" not in training_environment(2, hide_gpu=False)


def test_training_environment_is_a_copy_of_the_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    before = dict(os.environ)

    training_environment(2, hide_gpu=True)["OMP_NUM_THREADS"] = "x"

    assert dict(os.environ) == before


@pytest.mark.parametrize("threads", [0, -1])
def test_training_environment_rejects_fewer_than_one_thread(threads: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        training_environment(threads, hide_gpu=True)


def test_run_training_rejects_fewer_than_one_thread_before_it_launches_anything(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_training(tiny_config, run_dir, threads=0)

    assert fake_eole.calls == []
    assert not run_dir.exists()


# ---- run_training: commands and environment -------------------------------------


def test_eole_commands_are_the_two_commands_run_training_runs(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    build_vocab, train = eole_commands(tiny_config)

    run_training(tiny_config, run_dir)

    assert [list(build_vocab), list(train)] == [argv for argv, _ in fake_eole.calls]


def test_eole_commands_name_the_config_and_count_the_whole_corpus(
    tiny_config: Path,
) -> None:
    build_vocab, train = eole_commands(tiny_config)

    assert Path(build_vocab[0]).parent == Path(sys.executable).parent
    assert build_vocab[1:] == (
        "build_vocab",
        "-config",
        str(tiny_config),
        "-n_sample",
        "-1",
    )
    assert train[1:] == ("train", "-config", str(tiny_config))
    assert train[0] == build_vocab[0]


def test_eole_commands_report_a_missing_eole_script_and_launch_nothing(
    monkeypatch: pytest.MonkeyPatch,
    fake_eole: FakeEole,
    tmp_path: Path,
    tiny_config: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))

    with pytest.raises(UpstreamError, match="eole"):
        eole_commands(tiny_config)

    assert fake_eole.calls == []


def test_the_public_checkpoint_name_is_the_file_eole_links_to_its_newest_checkpoint() -> (
    None
):
    assert train_module.CHECKPOINT_FILE == CHECKPOINT_FILE


def test_run_training_runs_build_vocab_then_train_with_the_config(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    run_training(tiny_config, run_dir)

    assert [Path(argv[0]).name for argv, _ in fake_eole.calls] == ["eole", "eole"]
    assert [argv[1:] for argv, _ in fake_eole.calls] == [
        ["build_vocab", "-config", str(tiny_config), "-n_sample", "-1"],
        ["train", "-config", str(tiny_config)],
    ]


def test_run_training_calls_the_eole_script_next_to_the_interpreter(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    run_training(tiny_config, run_dir)

    interpreter_dir = Path(sys.executable).parent
    assert {Path(argv[0]).parent for argv, _ in fake_eole.calls} == {interpreter_dir}


def test_run_training_sets_the_thread_variables_and_hides_the_gpu_of_a_cpu_config(
    monkeypatch: pytest.MonkeyPatch,
    fake_eole: FakeEole,
    tiny_config: Path,
    run_dir: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    run_training(tiny_config, run_dir, threads=3)

    for _, kwargs in fake_eole.calls:
        assert kwargs["env"]["OMP_NUM_THREADS"] == "3"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "3"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_run_training_leaves_the_gpu_visible_for_a_config_that_trains_on_one(
    monkeypatch: pytest.MonkeyPatch,
    fake_eole: FakeEole,
    tmp_path: Path,
    run_dir: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    cfg_path, _ = _write_config(tmp_path, run_dir, profile="full")

    run_training(cfg_path, run_dir)

    assert [kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in fake_eole.calls] == [
        "1",
        "1",
    ]


def test_run_training_defaults_to_two_threads(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    run_training(tiny_config, run_dir)

    assert fake_eole.calls[0][1]["env"]["OMP_NUM_THREADS"] == "2"


def test_run_training_does_not_bound_the_commands_unless_asked(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    run_training(tiny_config, run_dir)

    assert fake_eole.waits == [None, None]


def test_run_training_bounds_each_command_by_the_timeout(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    run_training(tiny_config, run_dir, timeout=7.5)

    assert fake_eole.waits == [7.5, 7.5]


# ---- run_training: results and logs ---------------------------------------------


def test_run_training_returns_the_checkpoint_directory_the_config_names(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    assert run_training(tiny_config, run_dir) == run_dir / "ckpt"


def test_run_training_follows_a_model_path_other_than_the_default_inside_the_run_directory(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    model_path = run_dir / "models" / "tiny"
    cfg["training"]["model_path"] = model_path.as_posix()
    dump_config(cfg, cfg_path)

    assert run_training(cfg_path, run_dir) == model_path
    assert (model_path / CHECKPOINT_FILE).is_file()


def test_run_training_accepts_relative_paths(
    monkeypatch: pytest.MonkeyPatch, fake_eole: FakeEole, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = _write_config(tmp_path, Path("run"))

    assert run_training(cfg_path, Path("run")) == Path("run") / "ckpt"


def test_run_training_creates_the_run_directory_and_writes_one_log_per_command(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.output = ("first line", "second line")

    run_training(tiny_config, run_dir)

    for log_name in ("build_vocab.log", "train.log"):
        assert (run_dir / log_name).read_text(encoding="utf-8") == (
            "first line\nsecond line\n"
        )


def test_run_training_combines_stdout_and_stderr_in_the_log(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    run_training(tiny_config, run_dir)

    kwargs = fake_eole.calls[0][1]
    assert kwargs["stderr"] == subprocess.STDOUT


# ---- run_training: the checkpoint is new ----------------------------------------


def test_run_training_fails_when_training_writes_no_checkpoint(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.writes_results = False

    with pytest.raises(UpstreamError, match="no new checkpoint under"):
        run_training(tiny_config, run_dir)


def test_run_training_fails_when_only_an_older_runs_checkpoint_is_there(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    _stale_checkpoint(run_dir / "ckpt")
    fake_eole.writes_results = False

    with pytest.raises(UpstreamError, match="no new checkpoint under"):
        run_training(tiny_config, run_dir)


def test_run_training_fails_when_the_checkpoint_directory_is_empty(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    (run_dir / "ckpt").mkdir(parents=True)
    fake_eole.writes_results = False

    with pytest.raises(UpstreamError, match="no new checkpoint under"):
        run_training(tiny_config, run_dir)


def test_run_training_fails_when_the_checkpoint_link_points_nowhere(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    (run_dir / "ckpt").mkdir(parents=True)
    (run_dir / "ckpt" / CHECKPOINT_FILE).symlink_to("step_9/missing")
    fake_eole.writes_results = False

    with pytest.raises(UpstreamError, match="no new checkpoint under"):
        run_training(tiny_config, run_dir)


def test_run_training_accepts_a_checkpoint_that_replaces_an_older_runs(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    _stale_checkpoint(run_dir / "ckpt")

    assert run_training(tiny_config, run_dir) == run_dir / "ckpt"


def test_run_training_accepts_a_checkpoint_stamped_just_inside_the_two_second_slack(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.checkpoint_age = 1.8

    assert run_training(tiny_config, run_dir) == run_dir / "ckpt"


def test_run_training_rejects_a_checkpoint_stamped_just_outside_the_two_second_slack(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.checkpoint_age = 2.3

    with pytest.raises(UpstreamError, match="no new checkpoint under"):
        run_training(tiny_config, run_dir)


# ---- run_training: the model path comes from the config -------------------------


@pytest.mark.parametrize(
    "model_path",
    [None, 5, ""],
    ids=["missing", "not a string", "empty"],
)
def test_run_training_rejects_a_config_without_a_usable_model_path(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path, model_path: object
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    if model_path is None:
        del cfg["training"]["model_path"]
    else:
        cfg["training"]["model_path"] = model_path
    dump_config(cfg, cfg_path)

    with pytest.raises(ConfigError, match=r"training\.model_path"):
        run_training(cfg_path, run_dir)

    assert fake_eole.calls == []
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "outside",
    ["elsewhere/ckpt", "run", "run/../elsewhere", "run_two/ckpt"],
    ids=["sibling", "the run directory itself", "escaping with ..", "prefix sibling"],
)
def test_run_training_rejects_a_model_path_outside_the_run_directory(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path, outside: str
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    cfg["training"]["model_path"] = (tmp_path / outside).as_posix()
    dump_config(cfg, cfg_path)

    with pytest.raises(ConfigError, match="inside the run directory"):
        run_training(cfg_path, run_dir)

    assert fake_eole.calls == []
    assert not run_dir.exists()


# ---- run_training: failures -----------------------------------------------------


def test_run_training_validates_the_config_before_it_launches_anything(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    cfg["model"]["embeddings"]["position_encoding_type"] = "Rotary"
    dump_config(cfg, cfg_path)

    with pytest.raises(ConfigError, match="position_encoding_type"):
        run_training(cfg_path, run_dir)

    assert fake_eole.calls == []
    assert not run_dir.exists()


def test_run_training_rejects_a_missing_config_file(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    with pytest.raises(ConfigError, match="not found"):
        run_training(tmp_path / "absent.yaml", run_dir)

    assert fake_eole.calls == []


@pytest.mark.parametrize("text", ["key: [unclosed", "- just\n- a list\n"])
def test_run_training_rejects_a_config_that_is_not_a_mapping_of_settings(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path, text: str
) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError):
        run_training(cfg_path, run_dir)

    assert fake_eole.calls == []


def test_run_training_rejects_a_dollar_sign_in_the_config_path(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    cfg_path, _ = _write_config(tmp_path, run_dir)
    dollar_path = cfg_path.with_name("$HOME.yaml")
    cfg_path.rename(dollar_path)

    with pytest.raises(ConfigError, match=r"\$"):
        run_training(dollar_path, run_dir)

    assert fake_eole.calls == []


def test_run_training_rejects_a_dollar_sign_in_the_run_directory(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    dollar_run_dir = tmp_path / "run$1"
    cfg_path, _ = _write_config(tmp_path, dollar_run_dir)

    with pytest.raises(ConfigError, match=r"\$"):
        run_training(cfg_path, dollar_run_dir)

    assert fake_eole.calls == []
    assert not dollar_run_dir.exists()


def test_run_training_rejects_a_dollar_sign_anywhere_in_the_config_text(
    fake_eole: FakeEole, tmp_path: Path, run_dir: Path
) -> None:
    # Eole expands `$VAR` in the whole file, so a data path with one is rewritten too.
    cfg_path, cfg = _write_config(tmp_path, run_dir)
    cfg["data"]["valid"]["path_src"] = "corpus/$CORPUS/valid.src"
    dump_config(cfg, cfg_path)

    with pytest.raises(ConfigError, match=r"\$"):
        run_training(cfg_path, run_dir)

    assert fake_eole.calls == []


def test_run_training_reports_a_failed_build_vocab_with_the_log_tail_and_stops(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.exit_codes = {"build_vocab": 2}
    fake_eole.output = [f"line {number}" for number in range(1, 31)]

    with pytest.raises(UpstreamError) as raised:
        run_training(tiny_config, run_dir)

    message = str(raised.value)
    assert "exit code 2" in message
    assert message.endswith("\n".join(f"line {number}" for number in range(11, 31)))
    assert "line 10\n" not in message
    assert fake_eole.subcommands == ["build_vocab"]


def test_run_training_reports_a_failed_train_with_the_log_tail(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.exit_codes = {"train": 1}
    fake_eole.output = ["Traceback", "boom"]

    with pytest.raises(UpstreamError) as raised:
        run_training(tiny_config, run_dir)

    message = str(raised.value)
    assert "train" in message
    assert str(run_dir / "train.log") in message
    assert message.endswith("Traceback\nboom")
    assert fake_eole.subcommands == ["build_vocab", "train"]


def test_run_training_reports_a_missing_eole_script_without_running_anything(
    monkeypatch: pytest.MonkeyPatch,
    fake_eole: FakeEole,
    tmp_path: Path,
    tiny_config: Path,
    run_dir: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))

    with pytest.raises(UpstreamError, match="eole"):
        run_training(tiny_config, run_dir)

    assert fake_eole.calls == []


# ---- predict --------------------------------------------------------------------

WORDS = ("abc", "cab", "bca")


@pytest.fixture(scope="module")
def spm_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real character model that knows the tags, a, b, c and the suffix gap."""
    prefix = tmp_path_factory.mktemp("predict_model") / "chars"
    return train_char_model([*WORDS, f"a{NNBSP}b"], prefix, vocab_size=64)


def _predict(
    tmp_path: Path,
    default_model: Path,
    *,
    lines: Sequence[str] = WORDS,
    **overrides: Any,
) -> Path:
    src_file = tmp_path / "words.src"
    src_file.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    arguments: dict[str, Any] = {
        "ckpt_dir": tmp_path / "run" / "ckpt",
        "src_file": src_file,
        "tag": TO_SCRIPT_TAG,
        "spm_model": default_model,
        "out_file": tmp_path / "out" / "pred.tgt",
    }
    arguments.update(overrides)
    return predict(arguments.pop("ckpt_dir"), arguments.pop("src_file"), **arguments)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").removesuffix("\n").split("\n")


def test_predict_side_paths_are_the_log_the_config_and_the_two_token_files() -> None:
    out_file = Path("work") / "pred.tgt"

    assert predict_side_paths(out_file) == (
        Path("work") / "pred.tgt.log",
        Path("work") / "pred.tgt.predict.yaml",
        Path("work") / "pred.tgt.tokens.src",
        Path("work") / "pred.tgt.tokens.out",
    )


def test_predict_gives_eole_no_transforms_so_the_checkpoint_cannot_choose_the_tag(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    # Eole copies the `valid` corpus's prefix into the checkpoint's transform configs
    # and lets them override the prediction config, so a `prefix` transform would add
    # the checkpoint's tag whatever tag is asked for.
    out_file = _predict(tmp_path, spm_model)

    _, config_path, tokens_in, tokens_out = predict_side_paths(out_file)
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "model_path": (tmp_path / "run" / "ckpt").as_posix(),
        "src": tokens_in.as_posix(),
        "output": tokens_out.as_posix(),
        "beam_size": 1,
        "batch_size": 64,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": [],
    }
    assert "transforms_configs" not in fake_eole.predict_inputs[0][0]


@pytest.mark.parametrize("tag", [TO_SCRIPT_TAG, TO_CYRILLIC_TAG])
def test_predict_starts_every_token_line_with_the_requested_tag(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path, tag: str
) -> None:
    _predict(tmp_path, spm_model, tag=tag)

    ((_, token_lines),) = fake_eole.predict_inputs
    assert token_lines == [
        f"{tag} {DUMMY_PREFIX} a b c",
        f"{tag} {DUMMY_PREFIX} c a b",
        f"{tag} {DUMMY_PREFIX} b c a",
    ]
    assert all(line.split(" ")[0] == tag for line in token_lines)


def test_predict_gives_a_blank_source_line_the_tag_alone(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    _predict(tmp_path, spm_model, lines=["abc", "", "cab"])

    ((_, token_lines),) = fake_eole.predict_inputs
    assert token_lines[1] == TO_SCRIPT_TAG


def test_predict_keeps_a_suffix_gap_as_a_token_of_its_own_through_both_files(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    # NNBSP is not a separator: Eole splits token lines at U+0020 only, and so must
    # `predict`, or the gap vanishes between a stem and its suffix.
    out_file = _predict(tmp_path, spm_model, lines=[f"a{NNBSP}b"])

    ((_, token_lines),) = fake_eole.predict_inputs
    assert token_lines == [f"{TO_SCRIPT_TAG} {DUMMY_PREFIX} a {NNBSP} b"]
    assert _lines(out_file) == [f"a{NNBSP}b"]


def test_predict_decodes_the_token_lines_eole_wrote_into_the_output_file(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    processor = sentencepiece.SentencePieceProcessor(model_file=str(spm_model))
    fake_eole.hypotheses = [
        f"{DUMMY_PREFIX} b c a",
        f"{DUMMY_PREFIX} a b c",
        f"{DUMMY_PREFIX} c a b",
    ]

    out_file = _predict(tmp_path, spm_model)

    assert _lines(out_file) == ["bca", "abc", "cab"]
    assert _lines(out_file) == [
        processor.decode(hypothesis.split(" ")) for hypothesis in fake_eole.hypotheses
    ]
    assert out_file.read_text(encoding="utf-8") == "bca\nabc\ncab\n"


def test_predict_turns_an_empty_hypothesis_into_an_empty_line(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    fake_eole.hypotheses = [
        "",
        f"{DUMMY_PREFIX} a b c",
        "",
        f"{DUMMY_PREFIX} c a b",
        "",
    ]

    out_file = _predict(tmp_path, spm_model, lines=["x", "abc", "y", "cab", "z"])

    assert out_file.read_text(encoding="utf-8") == "\nabc\n\ncab\n\n"


def test_predict_writes_one_output_line_per_source_line(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    out_file = _predict(tmp_path, spm_model, lines=["abc", "", "cab", "bca"])

    assert _lines(out_file) == ["abc", "", "cab", "bca"]


def test_predict_reports_an_output_with_another_line_count_than_the_source(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    fake_eole.hypotheses = [f"{DUMMY_PREFIX} a b c", f"{DUMMY_PREFIX} c a b"]

    with pytest.raises(UpstreamError, match="2 lines for 3 source lines"):
        _predict(tmp_path, spm_model)


def test_predict_runs_eole_predict_on_the_cpu_and_returns_the_output_file(
    monkeypatch: pytest.MonkeyPatch,
    fake_eole: FakeEole,
    tmp_path: Path,
    spm_model: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    out_file = _predict(tmp_path, spm_model, threads=4)

    assert out_file == tmp_path / "out" / "pred.tgt"
    ((argv, kwargs),) = fake_eole.calls
    _, config_path, _, _ = predict_side_paths(out_file)
    assert Path(argv[0]).name in EOLE_NAMES
    assert argv[1:] == ["predict", "-config", str(config_path)]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
    assert kwargs["env"]["MKL_NUM_THREADS"] == "4"


def test_predict_bounds_the_command_by_the_timeout_only_when_asked(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    _predict(tmp_path, spm_model)
    _predict(tmp_path, spm_model, timeout=3.5)

    assert fake_eole.waits == [None, 3.5]


def test_predict_writes_the_combined_log_next_to_the_output_file(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    fake_eole.output = ("decoding",)

    out_file = _predict(tmp_path, spm_model)

    log_path, _, _, _ = predict_side_paths(out_file)
    assert log_path.read_text(encoding="utf-8") == "decoding\n"


def test_predict_reports_a_failure_with_the_log_tail(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    fake_eole.exit_codes = {"predict": 1}
    fake_eole.output = ["bad model"]

    with pytest.raises(UpstreamError) as raised:
        _predict(tmp_path, spm_model)

    assert "exit code 1" in str(raised.value)
    assert str(raised.value).endswith("bad model")


def test_predict_fails_when_eole_exits_zero_without_an_output_file(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    fake_eole.writes_results = False

    with pytest.raises(UpstreamError, match="no output file"):
        _predict(tmp_path, spm_model)


def test_predict_does_not_mistake_outputs_of_an_earlier_run_for_new_ones(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    out_file = tmp_path / "out" / "pred.tgt"
    _, _, _, tokens_out = predict_side_paths(out_file)
    out_file.parent.mkdir()
    out_file.write_text("left from an earlier run\n", encoding="utf-8")
    tokens_out.write_text("left from an earlier run\n", encoding="utf-8")
    fake_eole.writes_results = False

    with pytest.raises(UpstreamError, match="no output file"):
        _predict(tmp_path, spm_model)

    assert not out_file.exists()


def test_predict_creates_the_directory_of_the_output_file(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    out_file = _predict(tmp_path, spm_model, out_file=tmp_path / "a" / "b" / "pred.tgt")

    assert out_file.parent.is_dir()


@pytest.mark.parametrize("field", ["ckpt_dir", "src_file", "spm_model", "out_file"])
def test_predict_rejects_a_dollar_sign_in_a_path_before_writing_anything(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path, field: str
) -> None:
    with pytest.raises(ConfigError, match=r"holds '\$'"):
        _predict(tmp_path, spm_model, **{field: tmp_path / "out" / "$x"})

    assert fake_eole.calls == []
    assert not (tmp_path / "out").exists()


def test_predict_rejects_a_tag_the_model_does_not_know_before_writing_anything(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    # An unknown tag would reach Eole as <unk> and silently convert in no direction.
    with pytest.raises(ConfigError, match="not a piece of the model"):
        _predict(tmp_path, spm_model, tag="<2xx_Latn>")

    assert fake_eole.calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("problem", ["missing", "directory"])
def test_predict_rejects_a_character_model_that_cannot_be_loaded(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path, problem: str
) -> None:
    bad_model = tmp_path / "absent.model"
    if problem == "directory":
        bad_model.mkdir()

    with pytest.raises(DataError, match="cannot load SentencePiece model"):
        _predict(tmp_path, spm_model, spm_model=bad_model)

    assert fake_eole.calls == []
    assert not (tmp_path / "out").exists()


def test_predict_rejects_a_source_file_without_lines(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    with pytest.raises(DataError, match="no lines"):
        _predict(tmp_path, spm_model, lines=[])

    assert fake_eole.calls == []
    assert not (tmp_path / "out").exists()


def test_predict_rejects_a_missing_source_file(
    fake_eole: FakeEole, tmp_path: Path, spm_model: Path
) -> None:
    with pytest.raises(DataError, match="cannot read"):
        _predict(tmp_path, spm_model, src_file=tmp_path / "absent.src")

    assert fake_eole.calls == []
