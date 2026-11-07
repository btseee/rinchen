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
import yaml

from rinchen.errors import ConfigError, UpstreamError
from rinchen.script_conv.config import dump_config, render_converter_config
from rinchen.script_conv.train import (
    predict,
    predict_side_paths,
    run_training,
    training_environment,
)
from tests.support.fake_process import FakeProcess

TO_SCRIPT_TAG = "<2mn_Mong>"
EOLE_NAMES = ("eole", "eole.exe")
CHECKPOINT_FILE = "model.00.safetensors"  # what Eole links to its newest checkpoint


class FakeEole:
    """Stands in for `subprocess.Popen`: records every launch, writes `output` lines
    to the log handle it is given and exits with `exit_codes[subcommand]` (default 0).
    A finished `train` leaves a checkpoint (Eole's layout: a step directory and a
    link to its file), a finished `predict` its output file, unless `writes_results`
    is false. `checkpoint_age` is how many seconds before now the written
    checkpoint's modification time is set."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.waits: list[float | None] = []
        self.exit_codes: dict[str, int] = {}
        self.output: Sequence[str] = ("a log line",)
        self.writes_results = True
        self.checkpoint_age = 0.0

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((list(argv), kwargs))
        subcommand = argv[1]
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
            Path(cfg["output"]).write_text("predicted\n", encoding="utf-8")

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


def test_run_training_allows_for_file_systems_with_coarse_timestamps(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.checkpoint_age = 1.0

    assert run_training(tiny_config, run_dir) == run_dir / "ckpt"


def test_run_training_does_not_accept_a_checkpoint_written_long_before_the_launch(
    fake_eole: FakeEole, tiny_config: Path, run_dir: Path
) -> None:
    fake_eole.checkpoint_age = 30.0

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


def _predict(tmp_path: Path, **overrides: Any) -> Path:
    arguments: dict[str, Any] = {
        "ckpt_dir": tmp_path / "run" / "ckpt",
        "src_file": tmp_path / "words.src",
        "tag": TO_SCRIPT_TAG,
        "spm_model": tmp_path / "converter_char.model",
        "out_file": tmp_path / "out" / "pred.tgt",
    }
    arguments.update(overrides)
    return predict(arguments.pop("ckpt_dir"), arguments.pop("src_file"), **arguments)


def test_predict_side_paths_are_the_log_and_the_predict_config_next_to_the_output() -> (
    None
):
    out_file = Path("work") / "pred.tgt"

    assert predict_side_paths(out_file) == (
        Path("work") / "pred.tgt.log",
        Path("work") / "pred.tgt.predict.yaml",
    )


def test_predict_writes_the_predict_config_next_to_the_output_file(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    out_file = _predict(tmp_path)

    _, config_path = predict_side_paths(out_file)
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "model_path": (tmp_path / "run" / "ckpt").as_posix(),
        "src": (tmp_path / "words.src").as_posix(),
        "output": out_file.as_posix(),
        "beam_size": 1,
        "batch_size": 64,
        "batch_type": "sents",
        "gpu_ranks": [],
        "world_size": 1,
        "transforms": ["sentencepiece", "prefix"],
        "transforms_configs": {
            "sentencepiece": {
                "src_subword_model": (tmp_path / "converter_char.model").as_posix(),
                "tgt_subword_model": (tmp_path / "converter_char.model").as_posix(),
            },
            "prefix": {"src_prefix": TO_SCRIPT_TAG, "tgt_prefix": ""},
        },
    }


def test_predict_puts_the_requested_tag_in_the_prefix_transform(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    out_file = _predict(tmp_path, tag="<2mn_Cyrl>")

    _, config_path = predict_side_paths(out_file)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg["transforms_configs"]["prefix"]["src_prefix"] == "<2mn_Cyrl>"


def test_predict_runs_eole_predict_on_the_cpu_and_returns_the_output_file(
    monkeypatch: pytest.MonkeyPatch, fake_eole: FakeEole, tmp_path: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    out_file = _predict(tmp_path, threads=4)

    assert out_file == tmp_path / "out" / "pred.tgt"
    ((argv, kwargs),) = fake_eole.calls
    _, config_path = predict_side_paths(out_file)
    assert Path(argv[0]).name in EOLE_NAMES
    assert argv[1:] == ["predict", "-config", str(config_path)]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
    assert kwargs["env"]["MKL_NUM_THREADS"] == "4"


def test_predict_bounds_the_command_by_the_timeout_only_when_asked(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    _predict(tmp_path)
    _predict(tmp_path, timeout=3.5)

    assert fake_eole.waits == [None, 3.5]


def test_predict_writes_the_combined_log_next_to_the_output_file(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    fake_eole.output = ("decoding",)

    out_file = _predict(tmp_path)

    log_path, _ = predict_side_paths(out_file)
    assert log_path.read_text(encoding="utf-8") == "decoding\n"


def test_predict_reports_a_failure_with_the_log_tail(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    fake_eole.exit_codes = {"predict": 1}
    fake_eole.output = ["bad model"]

    with pytest.raises(UpstreamError) as raised:
        _predict(tmp_path)

    assert "exit code 1" in str(raised.value)
    assert str(raised.value).endswith("bad model")


def test_predict_fails_when_eole_exits_zero_without_an_output_file(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    fake_eole.writes_results = False

    with pytest.raises(UpstreamError, match="no output file"):
        _predict(tmp_path)


def test_predict_does_not_mistake_an_earlier_output_for_a_new_one(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    out_file = tmp_path / "out" / "pred.tgt"
    out_file.parent.mkdir()
    out_file.write_text("left from an earlier run\n", encoding="utf-8")
    fake_eole.writes_results = False

    with pytest.raises(UpstreamError, match="no output file"):
        _predict(tmp_path)


def test_predict_creates_the_directory_of_the_output_file(
    fake_eole: FakeEole, tmp_path: Path
) -> None:
    out_file = _predict(tmp_path, out_file=tmp_path / "a" / "b" / "pred.tgt")

    assert out_file.parent.is_dir()


@pytest.mark.parametrize(
    "field", ["ckpt_dir", "src_file", "spm_model", "out_file", "tag"]
)
def test_predict_rejects_a_dollar_sign_in_a_path_or_the_tag(
    fake_eole: FakeEole, tmp_path: Path, field: str
) -> None:
    # Eole expands `$VAR` in the whole prediction config, tag included.
    bad = "<2$tag>" if field == "tag" else tmp_path / "out" / "$x"

    with pytest.raises(ConfigError, match=r"\$"):
        _predict(tmp_path, **{field: bad})

    assert fake_eole.calls == []
    assert not (tmp_path / "out").exists()
